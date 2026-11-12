"""Batch C runtime integration for direct reduced-coordinate assembly."""

from __future__ import annotations

import functools
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from scipy import sparse

from . import nonlinear_performance as _performance
from . import nonlinear_performance_batch_b as _batch_b
from . import nonlinear_reduced_assembly as _reduced
from .jit_compiler import JIT_ENABLED
from .nonlinear_analysis_diagnostics import record_nonlinear_assembly_execution
from .nonlinear_element_evaluation import evaluate_nonlinear_element
from .nonlinear_state import NonlinearStateStore, discard_active_state_candidate
from .nonlinear_reduced_assembly import (
    ReducedAssemblyPlan,
    ReducedAssemblyPlanLimit,
    _identity_transformation,
    _maximum_map_bytes,
    assemble_reduced_system,
    build_reduced_assembly_plan,
)


@dataclass(frozen=True)
class _ReducedVectorPayload:
    values: np.ndarray
    token: object
    source_plan: Any
    assembly_call: int


@dataclass(frozen=True)
class _ReducedMatrixPayload:
    matrix: sparse.csr_matrix
    token: object


@dataclass(frozen=True)
class _ReducedLeftProduct:
    matrix: sparse.csr_matrix
    token: object

    def __matmul__(self, other: Any):
        if (
            isinstance(other, _DirectReductionTransform)
            and other._batch_c_token is self.token
        ):
            return self.matrix
        return NotImplemented


class _DirectReductionTranspose(sparse.csc_matrix):
    """Sparse transpose that recognizes already-reduced assembly payloads."""

    def __init__(
        self,
        arg1: Any,
        *,
        token: Optional[object] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(arg1, **kwargs)
        self._batch_c_token = token

    def __matmul__(self, other: Any):
        if (
            isinstance(other, _ReducedVectorPayload)
            and other.token is self._batch_c_token
        ):
            return other.values
        if (
            isinstance(other, _ReducedMatrixPayload)
            and other.token is self._batch_c_token
        ):
            return _ReducedLeftProduct(other.matrix, self._batch_c_token)
        return sparse.csc_matrix(self).__matmul__(other)


class _DirectReductionTransform(sparse.csr_matrix):
    """CSR transformation retaining a token for direct-reduction interception."""

    def __init__(
        self,
        arg1: Any,
        *,
        token: Optional[object] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(arg1, **kwargs)
        self._batch_c_token = token

    def transpose(self, axes=None, copy: bool = False):
        base = sparse.csr_matrix(self).transpose(axes=axes, copy=copy)
        return _DirectReductionTranspose(base, token=self._batch_c_token)


@dataclass
class _SolveContext:
    requested_model: Any
    token: object = field(default_factory=object)
    bound_model: Any = None
    transformation: Optional[_DirectReductionTransform] = None
    reduced_plan: Optional[ReducedAssemblyPlan] = None
    fallback_reason: Optional[str] = None
    estimated_assemblies: int = 0
    activation_threshold: int = 0
    activation_allowed: bool = False
    activation_reason: str = "not_evaluated"
    constraint_fallback_reason: Optional[str] = None
    direct_assembly_count: int = 0
    direct_residual_only_count: int = 0
    direct_plan_build_count: int = 0


_CONTEXT = threading.local()
_STATUS_LOCK = threading.RLock()
_STATUS: Dict[str, Any] = {
    "installed": False,
    "contexts_entered": 0,
    "reduced_plan_builds": 0,
    "reduced_assemblies": 0,
    "full_coordinate_fallbacks": 0,
    "cost_gate_skips": 0,
    "last_cost_gate": None,
    "last_fallback_reason": None,
    "last_plan": None,
}
_PATCHES: List[Tuple[ModuleType, str, Any, Any]] = []
_INSTALLED = False
_BASE_ASSEMBLER = None
_ORIGINAL_STATIC_SOLVER = None
_ORIGINAL_ARC_SOLVER = None
_ORIGINAL_CONSTRAINT_BUILDER = None
_ORIGINAL_LOCAL_EVALUATOR = _reduced._evaluate_local_responses


def _context_stack() -> List[_SolveContext]:
    stack = getattr(_CONTEXT, "stack", None)
    if stack is None:
        stack = []
        _CONTEXT.stack = stack
    return stack


def _active_context(model: Any) -> Optional[_SolveContext]:
    for context in reversed(_context_stack()):
        if context.bound_model is model:
            return context
    return None


def _minimum_estimated_assemblies() -> int:
    raw = os.environ.get("FE_SOLVER_BATCH_C_MIN_ESTIMATED_ASSEMBLIES", "144")
    try:
        return max(int(raw), 0)
    except ValueError:
        return 144


def _estimated_solver_assemblies(original: Any, args: Tuple[Any, ...], kwargs: Mapping[str, Any]) -> int:
    """Conservative useful-assembly estimate for the direct-reduction gate.

    The profiled MPC break-even was about 72 tangent evaluations.  Four
    evaluations per converged increment is deliberately more realistic than
    multiplying by the full Newton failure limit; the default activation
    threshold adds a second 2x safety margin.
    """

    max_iterations = int(kwargs.get("max_iterations", args[5] if len(args) > 5 else 25))
    typical_per_step = max(1, min(max_iterations, 4))
    if getattr(original, "__module__", "").endswith("arc_length"):
        control = kwargs.get("control")
        steps = int(getattr(control, "max_steps", 100))
    else:
        steps = int(kwargs.get("num_steps", args[4] if len(args) > 4 else 10))
    return max(steps, 0) * typical_per_step


def _has_nonzero_affine_path(model: Any) -> bool:
    for boundary in getattr(model, "boundary_conditions", ()) or ():
        for value in getattr(boundary, "dof_constraints", {}).values():
            try:
                if abs(float(value)) > 1.0e-14:
                    return True
            except (TypeError, ValueError):
                continue
    for equation in getattr(model, "constraint_equations", ()) or ():
        try:
            if abs(float(getattr(equation, "rhs", 0.0))) > 1.0e-14:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _run_with_context(original, *args: Any, **kwargs: Any):
    model = args[0] if args else kwargs.get("model")
    estimated = _estimated_solver_assemblies(original, args, kwargs)
    threshold = _minimum_estimated_assemblies()
    allowed = estimated >= threshold
    prescribed_arc_path = (
        getattr(original, "__module__", "").endswith("arc_length")
        and _has_nonzero_affine_path(model)
    )
    if prescribed_arc_path:
        # Direct reduced assembly intentionally does not retain the full
        # constrained/free tangent coupling K*u0 required by the derivative
        # of u = T*q + lambda*u0.  Use the qualified full-coordinate path.
        allowed = False
    context = _SolveContext(
        requested_model=model,
        estimated_assemblies=estimated,
        activation_threshold=threshold,
        activation_allowed=allowed,
        activation_reason=(
            "prescribed_arc_path_requires_full_tangent"
            if prescribed_arc_path
            else (
                "estimated_assembly_budget_meets_threshold"
                if allowed
                else "estimated_assembly_budget_below_threshold"
            )
        ),
    )
    stack = _context_stack()
    stack.append(context)
    with _STATUS_LOCK:
        _STATUS["contexts_entered"] += 1
        _STATUS["last_cost_gate"] = {
            "estimated_assemblies": estimated,
            "activation_threshold": threshold,
            "activated": allowed,
            "reason": context.activation_reason,
        }
        if not allowed:
            _STATUS["cost_gate_skips"] += 1
    try:
        return original(*args, **kwargs)
    finally:
        if stack and stack[-1] is context:
            stack.pop()
        else:
            try:
                stack.remove(context)
            except ValueError:
                pass


def _make_solver_wrapper(original):
    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any):
        return _run_with_context(original, *args, **kwargs)

    wrapper._batch_c_original = original
    return wrapper


def _constraint_builder_wrapper(K, F, model):
    result = _ORIGINAL_CONSTRAINT_BUILDER(K, F, model)
    K_red, F_red, transformation, u0, independent_dofs, info = result
    stack = _context_stack()
    if not stack:
        return result
    context = stack[-1]
    if context.bound_model is None:
        context.bound_model = model
    if (
        context.bound_model is not model
        or not context.activation_allowed
    ):
        return result
    if int(transformation.shape[1]) == 0:
        context.constraint_fallback_reason = "empty_reduced_system"
        return result
    if _identity_transformation(transformation):
        context.constraint_fallback_reason = "identity_constraint_transformation"
        return result
    wrapped = _DirectReductionTransform(transformation, token=context.token)
    context.transformation = wrapped
    return K_red, F_red, wrapped, u0, independent_dofs, info


def _batch_c_evaluate_local_responses(
    nonlinear_plan: Any,
    displacements: np.ndarray,
    committed_states: Mapping[int, Any],
    tangent: bool,
):
    """Fill local buffers while retaining Batch B's elastic shell fast path."""

    start = time.perf_counter()
    nonlinear_plan.force_values.fill(0.0)
    if tangent:
        nonlinear_plan.tangent_values.fill(0.0)
    trial_states: Dict[int, Any] = {}
    displacement_array = np.asarray(displacements, dtype=float)
    state_token = (
        committed_states.active_trial_token()
        if isinstance(committed_states, NonlinearStateStore)
        else None
    )

    for batch in nonlinear_plan.shell_batches:
        used_persistent_state = False
        if getattr(batch, "_batch_b_elastic", False):
            initialized_count = _batch_b._populate_initial_field_resultants(
                batch, committed_states
            )
            kernel_start = time.perf_counter()
            _batch_b._elastic_shell_batch_into_buffers(
                displacement_array,
                batch.dof_mappings,
                batch.T0,
                batch.B_m,
                batch.B_b,
                batch.B_d,
                batch.Gw,
                batch.detw,
                batch.B_s,
                batch.detw_shear,
                batch._batch_b_membrane_matrix,
                batch._batch_b_coupling_matrix,
                batch._batch_b_bending_matrix,
                batch._batch_b_initial_membrane_resultants,
                batch._batch_b_initial_bending_resultants,
                batch._batch_b_shear_matrix,
                batch._batch_b_drilling_stiffness,
                batch.force_positions,
                batch.tangent_positions,
                nonlinear_plan.force_values,
                nonlinear_plan.tangent_values,
                batch.u_work,
                bool(tangent),
            )
            if np.any(batch.linear_corrections):
                element_displacements = displacement_array[batch.dof_mappings]
                correction_forces = np.einsum(
                    "eij,ej->ei",
                    batch.linear_corrections,
                    element_displacements,
                    optimize=True,
                )
                nonlinear_plan.force_values[batch.force_positions] += (
                    correction_forces
                )
                if tangent:
                    nonlinear_plan.tangent_values[batch.tangent_positions] += (
                        batch.linear_corrections.reshape(len(batch.elements), -1)
                    )
            nonlinear_plan.timings.shell_kernel_seconds += (
                time.perf_counter() - kernel_start
            )
            nonlinear_plan.timings.initial_field_accelerated_elements += initialized_count
            if getattr(batch, "_batch_b_generalized", False):
                trial_states.update(_batch_b._generalized_trial_states(batch))
            else:
                elastic_states = batch._batch_b_elastic_state_mapping
                use_cached_mapping = True
                for element_id in batch.element_ids:
                    existing = committed_states.get(int(element_id))
                    if (
                        isinstance(existing, dict)
                        and existing is not elastic_states[int(element_id)]
                    ):
                        use_cached_mapping = False
                        break
                if use_cached_mapping:
                    trial_states.update(elastic_states)
                else:
                    for element_id in batch.element_ids:
                        element_key = int(element_id)
                        existing = committed_states.get(element_key)
                        trial_states[element_key] = (
                            existing
                            if isinstance(existing, dict)
                            else elastic_states[element_key]
                        )
            if initialized_count and not tangent:
                _batch_b._recover_initial_field_states(
                    nonlinear_plan,
                    batch,
                    displacement_array,
                    committed_states,
                    trial_states,
                )
        else:
            persistent_trial = (
                committed_states.shell_trial_for_layout(
                    state_token,
                    batch.state_layout,
                )
                if isinstance(committed_states, NonlinearStateStore)
                and state_token is not None
                else None
            )
            if persistent_trial is not None and batch.persistent_state_eligibility(
                persistent_trial[0]
            )[0]:
                used_persistent_state = True
                force_batch, tangent_batch, kernel_seconds = batch.evaluate_persistent(
                    displacement_array,
                    persistent_trial[0],
                    persistent_trial[1],
                    tangent,
                )
                batch_states = {}
            else:
                force_batch, tangent_batch, batch_states, kernel_seconds = batch.evaluate(
                    displacement_array,
                    committed_states,
                    tangent,
                )
            nonlinear_plan.timings.shell_kernel_seconds += kernel_seconds
            nonlinear_plan.force_values[batch.force_positions.reshape(-1)] = np.asarray(
                force_batch,
                dtype=float,
            ).reshape(-1)
            if tangent and tangent_batch is not None:
                nonlinear_plan.tangent_values[
                    batch.tangent_positions.reshape(-1)
                ] = np.asarray(tangent_batch, dtype=float).reshape(-1)
            trial_states.update(batch_states)

        if not used_persistent_state:
            override_start = time.perf_counter()
            override_count = _performance._apply_initial_field_shell_overrides(
                nonlinear_plan,
                batch,
                displacement_array,
                committed_states,
                tangent,
                trial_states,
            )
            nonlinear_plan.timings.initial_field_override_seconds += (
                time.perf_counter() - override_start
            )
            nonlinear_plan.timings.initial_field_override_elements += override_count

    for batch in nonlinear_plan.quadratic_beam_batches:
        nonlinear_plan.timings.non_shell_seconds += batch.evaluate_into_buffers(
            nonlinear_plan,
            displacement_array,
            committed_states,
            tangent,
            trial_states,
        )

    non_shell_start = time.perf_counter()
    model = nonlinear_plan.model
    mesh = model.mesh
    for record in nonlinear_plan.non_shell_elements:
        material = model.get_material(record.element.material_name)
        element_displacement = displacement_array[record.dof_mapping]
        force_element, tangent_element, trial_state = (
            evaluate_nonlinear_element(
                record.element,
                mesh,
                material,
                element_displacement,
                committed_states.get(record.element_id),
                nonlinear_plan.num_layers,
                tangent,
                committed_states=committed_states,
                state_token=state_token,
                element_id=record.element_id,
                kinematics="von_karman",
            )
        )
        nonlinear_plan.force_values[record.force_positions] = np.asarray(
            force_element,
            dtype=float,
        ).reshape(-1)
        if tangent and tangent_element is not None:
            nonlinear_plan.tangent_values[record.tangent_positions] = np.asarray(
                tangent_element,
                dtype=float,
            ).reshape(-1)
        if trial_state is not None:
            trial_states[record.element_id] = trial_state
    nonlinear_plan.timings.non_shell_seconds += (
        time.perf_counter() - non_shell_start
    )
    return trial_states, time.perf_counter() - start


def _batch_c_assemble_nonlinear_system(
    model,
    displacements: np.ndarray,
    committed_states: Dict[int, Any],
    num_layers: int,
    tangent: bool = True,
    deleted_element_ids=None,
    residual_stiffness_fraction: float = 1.0,
    **extra,
):
    deleted_tuple = tuple(deleted_element_ids or ())
    require_full_coordinates = bool(extra.pop("require_full_coordinates", False))
    kinematics = str(extra.pop("kinematics", "von_karman"))
    corotational_tangent = str(
        extra.pop("corotational_tangent", "rotated")
    )
    # Direct reduced assembly returns a residual/tangent without visiting the
    # reference assembler's contact-contributor hook.  Keep contact candidates
    # on that transactional full-coordinate path until a reduced contact
    # assembly is derived and checked against it.
    from .general_contact.runtime import active_contact_manager

    if (active_contact_manager() is not None or require_full_coordinates
            or deleted_tuple or extra or kinematics != "von_karman"):
        # The reduced-assembly plan encodes the default von Karman element
        # response; erosion, per-element stiffness scales, corotational
        # kinematics take the full reference path. Immutable initial fields
        # remain eligible through exact local response overrides.
        return _BASE_ASSEMBLER(
            model,
            displacements,
            committed_states,
            num_layers,
            tangent=tangent,
            deleted_element_ids=deleted_tuple,
            residual_stiffness_fraction=float(residual_stiffness_fraction),
            kinematics=kinematics,
            corotational_tangent=corotational_tangent,
            **extra,
        )
    context = _active_context(model)
    if (
        context is None
        or context.transformation is None
        or context.transformation.shape[1] == 0
        or context.fallback_reason is not None
    ):
        return _BASE_ASSEMBLER(
            model,
            displacements,
            committed_states,
            num_layers,
            tangent=tangent,
            deleted_element_ids=deleted_tuple,
            residual_stiffness_fraction=float(residual_stiffness_fraction),
        )

    assembly_start = time.perf_counter()
    from .nonlinear_performance_bootstrap import get_nonlinear_assembly_plan

    nonlinear_plan = get_nonlinear_assembly_plan(model, int(num_layers))
    if (
        context.reduced_plan is None
        or context.reduced_plan.source_plan is not nonlinear_plan
    ):
        try:
            context.reduced_plan = build_reduced_assembly_plan(
                nonlinear_plan,
                context.transformation,
            )
            context.direct_plan_build_count += 1
            with _STATUS_LOCK:
                _STATUS["reduced_plan_builds"] += 1
                _STATUS["last_plan"] = context.reduced_plan.diagnostics()
        except ReducedAssemblyPlanLimit as exc:
            context.fallback_reason = str(exc)
            with _STATUS_LOCK:
                _STATUS["full_coordinate_fallbacks"] += 1
                _STATUS["last_fallback_reason"] = context.fallback_reason
            return _BASE_ASSEMBLER(
                model,
                displacements,
                committed_states,
                num_layers,
                tangent=tangent,
                deleted_element_ids=deleted_tuple,
                residual_stiffness_fraction=float(residual_stiffness_fraction),
            )

    # Keep the assembly and lazy force-generation capture in one re-entrant
    # plan-lock scope.  Otherwise a same-model assembly could replace the
    # local buffers after ``assemble_reduced_system`` released its inner lock
    # but before this payload recorded ``timings.calls``.
    try:
        with nonlinear_plan._lock:
            force_reduced, tangent_reduced, trial_states = assemble_reduced_system(
                nonlinear_plan,
                context.reduced_plan,
                displacements,
                committed_states,
                tangent=tangent,
            )
            force_payload = _ReducedVectorPayload(
                force_reduced,
                context.token,
                nonlinear_plan,
                int(nonlinear_plan.timings.calls),
            )
        context.direct_assembly_count += 1
        if not tangent:
            context.direct_residual_only_count += 1
        with _STATUS_LOCK:
            _STATUS["reduced_assemblies"] += 1
            _STATUS["last_plan"] = context.reduced_plan.diagnostics()
        record_nonlinear_assembly_execution(
            path="direct_reduced",
            tangent=bool(tangent),
            elapsed_seconds=time.perf_counter() - assembly_start,
            plan=nonlinear_plan,
        )
        tangent_payload = (
            _ReducedMatrixPayload(tangent_reduced, context.token)
            if tangent_reduced is not None
            else None
        )
        return force_payload, tangent_payload, trial_states
    except BaseException:
        # Direct-reduced assembly owns the same candidate transaction as the
        # full-coordinate paths.  Never leak it across a failed local kernel
        # or post-assembly observer.
        discard_active_state_candidate(committed_states)
        raise


def materialize_full_internal_force(
    force_payload: Any,
    total_dofs: int,
) -> Optional[np.ndarray]:
    """Recover a full force without repeating an accepted local response.

    Ordinary full-coordinate assembly payloads are returned as one-dimensional
    arrays.  A direct-reduced payload is expanded from its element-local force
    buffers only while it is still the most recent assembly of that plan.  The
    generation check makes the optimization conservative when the same model
    is (unsupportedly) shared by overlapping solves: callers receive ``None``
    and may perform a fresh full-coordinate recovery instead of consuming
    buffers that another assembly has replaced.
    """

    expected = int(total_dofs)
    if isinstance(force_payload, _ReducedVectorPayload):
        plan = force_payload.source_plan
        with plan._lock:
            if (
                int(plan.total_dofs) != expected
                or int(plan.timings.calls) != int(force_payload.assembly_call)
            ):
                return None
            return _performance._scatter_sum(
                plan.force_values,
                plan.force_dofs_flat,
                plan.total_dofs,
            )

    try:
        full = np.asarray(force_payload, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if full.size != expected:
        return None
    return full


def _record_patch(module: ModuleType, name: str, replacement: Any) -> None:
    current = getattr(module, name, None)
    if current is replacement:
        return
    _PATCHES.append((module, name, current, replacement))
    setattr(module, name, replacement)


def _replace_function_references(original: Any, replacement: Any) -> None:
    """Replace already-imported aliases of a solver entry point.

    Batch C is installed lazily, so callers can legitimately bind
    ``solve_static_nonlinear`` or ``solve_static_arc_length`` before the first
    nonlinear solve triggers installation.  Restricting this pass to
    ``anyfesolver.*`` modules leaves those consumer aliases pointing at the
    unwrapped function and therefore bypasses the solve context required for
    direct reduced assembly.

    Only attributes whose value is the exact function object are changed, and
    every change is recorded in ``_PATCHES`` for conditional restoration by
    :func:`uninstall_batch_c_optimizations`.
    """

    for module in list(sys.modules.values()):
        if not isinstance(module, ModuleType):
            continue
        module_name = getattr(module, "__name__", "")
        if module_name == __name__:
            continue
        for name, value in list(vars(module).items()):
            if value is original:
                _record_patch(module, name, replacement)


def install_batch_c_optimizations() -> bool:
    """Install direct reduced-coordinate assembly for Numba-enabled runs."""

    global _INSTALLED, _BASE_ASSEMBLER, _ORIGINAL_STATIC_SOLVER
    global _ORIGINAL_ARC_SOLVER, _ORIGINAL_CONSTRAINT_BUILDER
    if _INSTALLED:
        return True
    if not JIT_ENABLED:
        return False

    from . import arc_length
    from . import assembly
    from . import nonlinear_static

    _BASE_ASSEMBLER = nonlinear_static._assemble_nonlinear_system
    _ORIGINAL_STATIC_SOLVER = nonlinear_static.solve_static_nonlinear
    _ORIGINAL_ARC_SOLVER = arc_length.solve_static_arc_length
    _ORIGINAL_CONSTRAINT_BUILDER = assembly.build_constraint_transformation

    static_wrapper = _make_solver_wrapper(_ORIGINAL_STATIC_SOLVER)
    arc_wrapper = _make_solver_wrapper(_ORIGINAL_ARC_SOLVER)
    _replace_function_references(_ORIGINAL_STATIC_SOLVER, static_wrapper)
    _replace_function_references(_ORIGINAL_ARC_SOLVER, arc_wrapper)
    _replace_function_references(
        _ORIGINAL_CONSTRAINT_BUILDER,
        _constraint_builder_wrapper,
    )

    for module in list(sys.modules.values()):
        if not isinstance(module, ModuleType):
            continue
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("anyfesolver") or module_name == __name__:
            continue
        if hasattr(module, "_assemble_nonlinear_system"):
            _record_patch(
                module,
                "_assemble_nonlinear_system",
                _batch_c_assemble_nonlinear_system,
            )

    _reduced._evaluate_local_responses = _batch_c_evaluate_local_responses
    _INSTALLED = True
    with _STATUS_LOCK:
        _STATUS["installed"] = True
    return True


def uninstall_batch_c_optimizations() -> None:
    """Restore references changed by :func:`install_batch_c_optimizations`."""

    global _INSTALLED
    if not _INSTALLED:
        return
    _reduced._evaluate_local_responses = _ORIGINAL_LOCAL_EVALUATOR
    for module, name, original, replacement in reversed(_PATCHES):
        if getattr(module, name, None) is replacement:
            setattr(module, name, original)
    _PATCHES.clear()
    _context_stack().clear()
    _INSTALLED = False
    with _STATUS_LOCK:
        _STATUS["installed"] = False


def reset_batch_c_counters() -> None:
    with _STATUS_LOCK:
        installed = bool(_STATUS["installed"])
        _STATUS.update(
            {
                "installed": installed,
                "contexts_entered": 0,
                "reduced_plan_builds": 0,
                "reduced_assemblies": 0,
                "full_coordinate_fallbacks": 0,
                "cost_gate_skips": 0,
                "last_cost_gate": None,
                "last_fallback_reason": None,
                "last_plan": None,
            }
        )


def current_batch_c_analysis_diagnostics() -> Dict[str, Any]:
    """Return selector diagnostics for the current thread-local solver call."""

    stack = _context_stack()
    if not stack:
        return {
            "context_active": False,
            "activated": False,
            "fallback_reason": "direct_reduction_context_not_active",
            "estimated_assemblies": 0,
            "activation_threshold": _minimum_estimated_assemblies(),
            "assembly_count": 0,
            "residual_only_assembly_count": 0,
            "plan_build_count": 0,
            "plan_reused": False,
            "plan_reuse_scope": "within_analysis",
            "plan": None,
        }
    context = stack[-1]
    plan_diagnostics = (
        None if context.reduced_plan is None else context.reduced_plan.diagnostics()
    )
    assembly_count = int(context.direct_assembly_count)
    activated = assembly_count > 0
    if activated and context.fallback_reason is not None:
        fallback_reason = "partial_full_coordinate_fallback"
    elif activated:
        fallback_reason = None
    else:
        fallback_reason = (
            context.fallback_reason
            or context.constraint_fallback_reason
            or (
                context.activation_reason
                if not context.activation_allowed
                else "direct_reduction_not_exercised"
            )
        )
    return {
        "context_active": True,
        "activated": activated,
        "fallback_reason": fallback_reason,
        "estimated_assemblies": int(context.estimated_assemblies),
        "activation_threshold": int(context.activation_threshold),
        "activation_reason": str(context.activation_reason),
        "assembly_count": assembly_count,
        "residual_only_assembly_count": int(context.direct_residual_only_count),
        "plan_build_count": int(context.direct_plan_build_count),
        "plan_reused": bool(assembly_count > context.direct_plan_build_count),
        "plan_reuse_scope": "within_analysis",
        "plan": plan_diagnostics,
    }


def batch_c_status() -> Dict[str, Any]:
    with _STATUS_LOCK:
        result = dict(_STATUS)
    result["eligible"] = bool(JIT_ENABLED)
    result["map_limit_mb"] = float(_maximum_map_bytes() / (1024.0**2))
    result["default_activation_threshold"] = _minimum_estimated_assemblies()
    result["active_context_depth"] = len(_context_stack())
    result["batch_b_local_kernel_retained"] = bool(
        _reduced._evaluate_local_responses is _batch_c_evaluate_local_responses
    )
    return result
