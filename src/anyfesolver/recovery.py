"""Selective result recovery and resource-policy helpers."""

from __future__ import annotations

import copy
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .element_capabilities import require_model_element_capabilities
from .jit_compiler import JIT_DISABLED_REASON, JIT_ENABLED, numba_thread_scope
from .recovery_batches import (
    RecoveryPlanItem,
    _get_recovery_batch_plan_under_lease,
    _run_with_qualified_recovery_runtime_lease,
    build_recovery_chunks,
    clear_recovery_batch_plan,
    formulation_counts,
    get_recovery_batch_plan,
)
from .recovery_s4 import recover_isotropic_s4
from .s3_reference_batch import (
    MIN_REFERENCE_S3_RECOVERY_GROUP,
    recover_reference_s3,
)
from .threading_policy import native_thread_scope

if TYPE_CHECKING:
    from .fe_core import FEModel


_DOF_COMPONENTS = ("ux", "uy", "uz", "rx", "ry", "rz")
_HISTORY_MODES = {"full", "selected", "envelope"}
_MIN_COMPILED_RECOVERY_BATCH = 100
_MIN_RECOVERY_PLAN_SIZE = 100


def _optional_int_tuple(values: Optional[Sequence[int]]) -> Optional[Tuple[int, ...]]:
    if values is None:
        return None
    return tuple(int(value) for value in values)


def _optional_str_tuple(values: Optional[Sequence[str]]) -> Optional[Tuple[str, ...]]:
    if values is None:
        return None
    return tuple(str(value) for value in values)


@dataclass(frozen=True)
class RecoveryConfig:
    """Requested result-recovery scope.

    ``None`` for node or element ids means recover all available items, matching
    the legacy result behavior.  Components filter stress/result dictionaries by
    key; displacement arrays remain six-DOF node vectors.
    """

    node_ids: Optional[Sequence[int]] = None
    element_ids: Optional[Sequence[int]] = None
    components: Optional[Sequence[str]] = None
    include_displacements: bool = True
    include_stresses: bool = True
    include_reactions: bool = True
    history_mode: str = "full"
    store_full_histories: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.history_mode not in _HISTORY_MODES:
            raise ValueError(f"history_mode must be one of {sorted(_HISTORY_MODES)}")

    def selected_node_ids(self, model: "FEModel") -> Tuple[int, ...]:
        if self.node_ids is None:
            return tuple(int(node_id) for node_id in model.mesh.nodes)
        missing = [int(node_id) for node_id in self.node_ids if int(node_id) not in model.mesh.nodes]
        if missing:
            raise ValueError(f"Requested recovery node ids not found: {missing}")
        return _optional_int_tuple(self.node_ids) or ()

    def selected_element_ids(self, model: "FEModel") -> Tuple[int, ...]:
        if self.element_ids is None:
            return tuple(int(element_id) for element_id in model.mesh.elements)
        missing = [int(element_id) for element_id in self.element_ids if int(element_id) not in model.mesh.elements]
        if missing:
            raise ValueError(f"Requested recovery element ids not found: {missing}")
        return _optional_int_tuple(self.element_ids) or ()

    def selected_components(self) -> Optional[Tuple[str, ...]]:
        return _optional_str_tuple(self.components)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_ids": None if self.node_ids is None else list(_optional_int_tuple(self.node_ids) or ()),
            "element_ids": None if self.element_ids is None else list(_optional_int_tuple(self.element_ids) or ()),
            "components": None if self.components is None else list(_optional_str_tuple(self.components) or ()),
            "include_displacements": bool(self.include_displacements),
            "include_stresses": bool(self.include_stresses),
            "include_reactions": bool(self.include_reactions),
            "history_mode": self.history_mode,
            "store_full_histories": bool(self.store_full_histories),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ResourceConfig:
    """Bounded resource policy for solver phases.

    ``solver_threads`` is enforced as a scoped native BLAS/OpenMP limit and is
    restored after success or failure. ``assembly_threads`` scopes Numba
    workers; when Numba assembly is explicitly parallel, inner native pools are
    restricted to one thread to prevent oversubscription. Omitted limits retain
    backend defaults.
    """

    solver_threads: Optional[int] = None
    assembly_threads: Optional[int] = None
    recovery_threads: Optional[int] = None
    process_workers: Optional[int] = None
    deterministic: bool = True
    memory_limit_bytes: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("solver_threads", "assembly_threads", "recovery_threads", "process_workers"):
            value = getattr(self, name)
            if value is not None and int(value) <= 0:
                raise ValueError(f"{name} must be positive when provided")
        if self.memory_limit_bytes is not None and int(self.memory_limit_bytes) <= 0:
            raise ValueError("memory_limit_bytes must be positive when provided")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "solver_threads": None if self.solver_threads is None else int(self.solver_threads),
            "assembly_threads": None if self.assembly_threads is None else int(self.assembly_threads),
            "recovery_threads": None if self.recovery_threads is None else int(self.recovery_threads),
            "process_workers": None if self.process_workers is None else int(self.process_workers),
            "deterministic": bool(self.deterministic),
            "memory_limit_bytes": None if self.memory_limit_bytes is None else int(self.memory_limit_bytes),
            "metadata": dict(self.metadata),
        }


def _owned_resource_config_snapshot(
    value: Optional[ResourceConfig],
    *,
    post_observation: Any,
) -> Optional[ResourceConfig]:
    """Detach resource policy data before a solver decorator observes it.

    Qualified operations pass a non-renewable authority check as
    ``post_observation``.  Keeping that check immediately after every caller
    protocol means a hostile-but-supported mapping or numeric conversion
    cannot mutate and restore qualified mechanics while authoring the policy
    subsequently consumed by :func:`resource_threaded`.
    """

    if value is None:
        return None

    def observed_attribute(name: str) -> Any:
        observed = getattr(value, name)
        post_observation()
        return observed

    def optional_int(name: str) -> Optional[int]:
        observed = observed_attribute(name)
        if observed is None:
            return None
        converted = int(observed)
        post_observation()
        return converted

    def owned_plain(observed: Any) -> Any:
        if observed is None or type(observed) in (bool, int, float, str, bytes):
            return observed
        if isinstance(observed, np.ndarray):
            detached = np.array(observed, copy=True)
            post_observation()
            return detached
        if isinstance(observed, Mapping):
            iterator = iter(observed)
            post_observation()
            detached_mapping: Dict[Any, Any] = {}
            while True:
                try:
                    key = next(iterator)
                except StopIteration:
                    post_observation()
                    break
                post_observation()
                detached_key = owned_plain(key)
                member = observed[key]
                post_observation()
                detached_member = owned_plain(member)
                detached_mapping[detached_key] = detached_member
                post_observation()
            return detached_mapping
        if isinstance(observed, Sequence):
            iterator = iter(observed)
            post_observation()
            detached_sequence = []
            while True:
                try:
                    member = next(iterator)
                except StopIteration:
                    post_observation()
                    break
                post_observation()
                detached_sequence.append(owned_plain(member))
            return tuple(detached_sequence)
        detached = copy.deepcopy(observed)
        post_observation()
        return detached

    deterministic_observed = observed_attribute("deterministic")
    deterministic = bool(deterministic_observed)
    post_observation()
    metadata = owned_plain(observed_attribute("metadata"))
    if not isinstance(metadata, Mapping):
        raise ValueError("resource metadata must be a mapping")
    owned = ResourceConfig(
        solver_threads=optional_int("solver_threads"),
        assembly_threads=optional_int("assembly_threads"),
        recovery_threads=optional_int("recovery_threads"),
        process_workers=optional_int("process_workers"),
        deterministic=deterministic,
        memory_limit_bytes=optional_int("memory_limit_bytes"),
        metadata=metadata,
    )
    post_observation()
    return owned


class ResourcePolicyError(ValueError):
    """Raised when a requested resource policy cannot be satisfied."""

    def __init__(
        self,
        message: str,
        *,
        context: str,
        memory_estimate: Optional["MemoryEstimate"] = None,
        resource_config: Optional[ResourceConfig] = None,
    ) -> None:
        super().__init__(message)
        self.context = context
        self.memory_estimate = memory_estimate
        self.resource_config = resource_config

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context": self.context,
            "message": str(self),
            "memory_estimate": None if self.memory_estimate is None else self.memory_estimate.to_dict(),
            "resource_config": None if self.resource_config is None else self.resource_config.to_dict(),
        }


@dataclass(frozen=True)
class MemoryEstimate:
    """Conservative byte estimates for common FE storage blocks."""

    total_dofs: int
    num_nodes: int
    num_elements: int
    matrix_nnz_estimate: int
    csr_bytes_estimate: int
    rhs_bytes_estimate: int
    history_bytes_estimate: int
    eigenvector_bytes_estimate: int
    nonlinear_state_bytes_estimate: int
    total_bytes_estimate: int
    notes: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_dofs": int(self.total_dofs),
            "num_nodes": int(self.num_nodes),
            "num_elements": int(self.num_elements),
            "matrix_nnz_estimate": int(self.matrix_nnz_estimate),
            "csr_bytes_estimate": int(self.csr_bytes_estimate),
            "rhs_bytes_estimate": int(self.rhs_bytes_estimate),
            "history_bytes_estimate": int(self.history_bytes_estimate),
            "eigenvector_bytes_estimate": int(self.eigenvector_bytes_estimate),
            "nonlinear_state_bytes_estimate": int(self.nonlinear_state_bytes_estimate),
            "total_bytes_estimate": int(self.total_bytes_estimate),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class RecoveryExecutionReport:
    """Execution diagnostics for a recovery phase."""

    phase: str
    item_count: int
    requested_workers: int
    used_workers: int
    backend: str
    deterministic: bool
    elapsed_seconds: float
    reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "item_count": int(self.item_count),
            "requested_workers": int(self.requested_workers),
            "used_workers": int(self.used_workers),
            "backend": self.backend,
            "deterministic": bool(self.deterministic),
            "elapsed_seconds": float(self.elapsed_seconds),
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class StressRecoveryProvenance:
    """Provenance for unified elastic/material-history stress recovery.

    ``per_element_source`` is deliberately explicit: callers can distinguish
    stresses reconstructed from total displacement (elastic) from stresses
    recovered from a committed shell-layer or beam-fiber material state.
    """

    mode: str
    state_source: str
    per_element_source: Mapping[int, str] = field(default_factory=dict)
    per_element_component_sources: Mapping[int, Mapping[str, str]] = field(
        default_factory=dict
    )
    history_aware_element_ids: Tuple[int, ...] = ()
    elastic_reconstruction_element_ids: Tuple[int, ...] = ()
    fallback_reasons: Mapping[int, str] = field(default_factory=dict)
    analysis_context: Mapping[str, Any] = field(default_factory=dict)
    return_global: bool = False
    warnings: Tuple[str, ...] = ()

    @property
    def history_aware(self) -> bool:
        return bool(self.history_aware_element_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": str(self.mode),
            "state_source": str(self.state_source),
            "per_element_source": {
                int(element_id): str(source)
                for element_id, source in sorted(self.per_element_source.items())
            },
            "per_element_component_sources": {
                int(element_id): {
                    str(component): str(source)
                    for component, source in sorted(component_sources.items())
                }
                for element_id, component_sources in sorted(
                    self.per_element_component_sources.items()
                )
            },
            "history_aware_element_ids": [int(value) for value in self.history_aware_element_ids],
            "elastic_reconstruction_element_ids": [
                int(value) for value in self.elastic_reconstruction_element_ids
            ],
            "fallback_reasons": {
                int(element_id): str(reason)
                for element_id, reason in sorted(self.fallback_reasons.items())
            },
            "analysis_context": dict(self.analysis_context),
            "return_global": bool(self.return_global),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class PatchRecoveryConfig:
    """Conservative settings for linear Zienkiewicz--Zhu-style shell patches.

    Patches are qualified only for consistently oriented, locally planar,
    homogeneous Q4 or Q8 shell neighborhoods.  Any failed topology,
    continuity, rank, or conditioning guard falls back to the established
    Gauss-to-node extrapolation and averaging method.
    """

    condition_limit: float = 1.0e8
    normal_tolerance_degrees: float = 15.0
    material_continuity_required: bool = True
    thickness_relative_tolerance: float = 1.0e-8
    planarity_relative_tolerance: float = 1.0e-3
    include_error_indicator: bool = False

    def __post_init__(self) -> None:
        if not np.isfinite(self.condition_limit) or float(self.condition_limit) <= 1.0:
            raise ValueError("condition_limit must be finite and greater than 1")
        if not 0.0 < float(self.normal_tolerance_degrees) < 90.0:
            raise ValueError("normal_tolerance_degrees must be in (0, 90)")
        if not np.isfinite(self.thickness_relative_tolerance) or float(self.thickness_relative_tolerance) < 0.0:
            raise ValueError("thickness_relative_tolerance must be finite and nonnegative")
        if not np.isfinite(self.planarity_relative_tolerance) or float(
            self.planarity_relative_tolerance
        ) < 0.0:
            raise ValueError(
                "planarity_relative_tolerance must be finite and nonnegative"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "polynomial_basis": "linear_2d",
            "condition_limit": float(self.condition_limit),
            "normal_tolerance_degrees": float(self.normal_tolerance_degrees),
            "material_continuity_required": bool(self.material_continuity_required),
            "thickness_relative_tolerance": float(self.thickness_relative_tolerance),
            "planarity_relative_tolerance": float(
                self.planarity_relative_tolerance
            ),
            "reduced_q8_qualified": False,
            "include_error_indicator": bool(self.include_error_indicator),
            "indicator_semantics": (
                "surface_stress_l2_discrepancy_not_energy_norm"
            ),
        }


@dataclass
class StressRecoveryResult:
    """Unified stress result with its committed state snapshot and provenance."""

    element_stresses: Dict[int, Dict[str, Any]]
    provenance: StressRecoveryProvenance
    committed_element_states: Dict[int, Any] = field(default_factory=dict, repr=False)
    execution_report: Optional[RecoveryExecutionReport] = None
    nodal_stresses: Optional[Dict[str, Any]] = None

    def get_element_stress(self, element_id: int) -> Optional[Dict[str, Any]]:
        return self.element_stresses.get(int(element_id))

    @property
    def history_aware(self) -> bool:
        return self.provenance.history_aware

    def provenance_dict(
        self, *, include_execution: bool = False
    ) -> Dict[str, Any]:
        """Return stable scientific provenance.

        Wall-clock timing is intentionally excluded by default so serialized
        result provenance is deterministic.  Callers that need operational
        diagnostics can request it explicitly or read ``execution_report``.
        """

        payload = self.provenance.to_dict()
        if include_execution and self.execution_report is not None:
            payload["execution"] = self.execution_report.to_dict()
        if self.nodal_stresses is not None:
            payload["nodal_recovery"] = {
                "method": self.nodal_stresses.get("method"),
                "qualified_node_count": len(self.nodal_stresses.get("qualified_node_ids", ())),
                "fallback_node_count": len(self.nodal_stresses.get("fallback_node_ids", ())),
                "error_indicator": self.nodal_stresses.get("error_indicator"),
            }
        return payload


def default_recovery_config(config: Optional[RecoveryConfig] = None) -> RecoveryConfig:
    """Return a full-recovery config when none is supplied."""

    return config if config is not None else RecoveryConfig()


def _estimate_matrix_nnz(model: "FEModel") -> int:
    """Assembled-matrix nonzero count, cached on the mesh topology revision.

    Only element connectivity determines the sparsity, so the (topological)
    count is cached and reused across the preflight and recovery memory checks
    of a solve, and across repeated solves on the same mesh.  The union of
    per-element DOF-pair blocks is computed vectorized rather than with a
    Python set of tuples.
    """
    mesh = model.mesh
    signature = mesh.revision_signature()
    cached = getattr(mesh, "_matrix_nnz_cache", None)
    if cached is not None and cached[0] == signature:
        return cached[1]
    total_dofs = int(mesh.dof_manager.total_dofs)
    encoded_blocks = []
    for element in mesh.elements.values():
        try:
            mapping = np.asarray(element.get_dof_mapping(mesh), dtype=np.int64).reshape(-1)
        except Exception:
            continue
        if mapping.size == 0:
            continue
        rows = np.repeat(mapping, mapping.size)
        cols = np.tile(mapping, mapping.size)
        encoded_blocks.append(rows * total_dofs + cols)
    if encoded_blocks:
        matrix_nnz = int(np.unique(np.concatenate(encoded_blocks)).size)
    else:
        matrix_nnz = 0
    mesh._matrix_nnz_cache = (signature, matrix_nnz)
    return matrix_nnz


def estimate_model_memory(
    model: "FEModel",
    *,
    num_rhs: int = 1,
    num_modes: int = 0,
    transient_saved_steps: int = 0,
    store_full_history: bool = True,
    recovery_config: Optional[RecoveryConfig] = None,
    nonlinear_state: bool = False,
    nonlinear_state_copies: int = 1,
) -> MemoryEstimate:
    """Estimate matrix/result storage for a model and recovery request."""

    if int(nonlinear_state_copies) <= 0:
        raise ValueError("nonlinear_state_copies must be positive")

    total_dofs = int(model.mesh.dof_manager.total_dofs)
    num_nodes = int(len(model.mesh.nodes))
    num_elements = int(len(model.mesh.elements))
    notes = []
    matrix_nnz = _estimate_matrix_nnz(model)
    if matrix_nnz == 0 and total_dofs:
        matrix_nnz = total_dofs
        notes.append("matrix sparsity estimated as diagonal because no element mapping was available")

    csr_bytes = int(matrix_nnz * (8 + 4) + (total_dofs + 1) * 4)
    rhs_bytes = int(max(int(num_rhs), 0) * total_dofs * 8)

    recovery = default_recovery_config(recovery_config)
    if transient_saved_steps > 0:
        if recovery.history_mode == "envelope":
            selected_history_dofs = 6 * len(recovery.selected_node_ids(model)) if recovery.node_ids is not None and recovery.include_displacements else 0
            history_bytes = int(total_dofs * 8 * 3 + max(int(transient_saved_steps), 0) * selected_history_dofs * 8)
        elif store_full_history and recovery.store_full_histories:
            history_dofs = total_dofs
            history_bytes = int(max(int(transient_saved_steps), 0) * max(history_dofs, 0) * 8 * 3)
        else:
            history_dofs = 6 * len(recovery.selected_node_ids(model)) if recovery.include_displacements else 0
            history_bytes = int(max(int(transient_saved_steps), 0) * max(history_dofs, 0) * 8 * 3)
    else:
        history_bytes = 0

    eigenvector_bytes = int(max(int(num_modes), 0) * total_dofs * 8)
    if nonlinear_state:
        nonlinear_bytes = int(
            max(num_elements, 0)
            * 8
            * 64
            * int(nonlinear_state_copies)
        )
        if int(nonlinear_state_copies) > 1:
            notes.append(
                "nonlinear state estimate includes "
                f"{int(nonlinear_state_copies)} retained state copies"
            )
    else:
        nonlinear_bytes = 0
    total = int(csr_bytes + rhs_bytes + history_bytes + eigenvector_bytes + nonlinear_bytes)
    return MemoryEstimate(
        total_dofs=total_dofs,
        num_nodes=num_nodes,
        num_elements=num_elements,
        matrix_nnz_estimate=matrix_nnz,
        csr_bytes_estimate=csr_bytes,
        rhs_bytes_estimate=rhs_bytes,
        history_bytes_estimate=history_bytes,
        eigenvector_bytes_estimate=eigenvector_bytes,
        nonlinear_state_bytes_estimate=nonlinear_bytes,
        total_bytes_estimate=total,
        notes=tuple(notes),
    )


def enforce_memory_limit(
    memory_estimate: MemoryEstimate,
    resource_config: Optional[ResourceConfig],
    *,
    context: str,
) -> None:
    """Raise when an estimate exceeds ``ResourceConfig.memory_limit_bytes``."""

    if resource_config is None or resource_config.memory_limit_bytes is None:
        return
    limit = int(resource_config.memory_limit_bytes)
    estimated = int(memory_estimate.total_bytes_estimate)
    if estimated > limit:
        raise ResourcePolicyError(
            f"{context} estimated memory {estimated} bytes exceeds configured limit {limit} bytes",
            context=context,
            memory_estimate=memory_estimate,
            resource_config=resource_config,
        )


def select_node_displacements(
    model: "FEModel",
    displacements: np.ndarray,
    recovery_config: Optional[RecoveryConfig] = None,
) -> Dict[int, np.ndarray]:
    """Extract selected nodal displacement vectors."""

    recovery = default_recovery_config(recovery_config)
    if not recovery.include_displacements:
        return {}
    vector = np.asarray(displacements, dtype=float)
    selected: Dict[int, np.ndarray] = {}
    for node_id in recovery.selected_node_ids(model):
        node = model.mesh.nodes[int(node_id)]
        selected[int(node_id)] = vector[np.asarray(node.dofs, dtype=np.intp)]
    return selected


def _filter_components(values: Mapping[str, Any], components: Optional[Tuple[str, ...]]) -> Dict[str, Any]:
    if components is None:
        return dict(values)
    wanted = set(components)
    return {key: value for key, value in values.items() if str(key) in wanted}


def _recovery_worker_count(resource_config: Optional[ResourceConfig], item_count: int) -> Tuple[int, int, str]:
    requested = 1 if resource_config is None or resource_config.recovery_threads is None else int(resource_config.recovery_threads)
    if item_count <= 1:
        return requested, 1, "serial: item count <= 1"
    if requested <= 1:
        return requested, 1, "serial: recovery_threads not requested"
    return requested, min(requested, int(item_count)), "thread_pool"


def _ordered_element_ids(model: "FEModel", selected: Sequence[int]) -> Tuple[int, ...]:
    wanted = {int(element_id) for element_id in selected}
    return tuple(int(element_id) for element_id in model.mesh.elements if int(element_id) in wanted)


def _compute_one_element_stress(
    model: "FEModel",
    displacements: np.ndarray,
    element_id: int,
    *,
    return_global: bool,
    dof_mapping: Optional[np.ndarray] = None,
) -> Optional[Tuple[int, Dict[str, np.ndarray]]]:
    element = model.mesh.elements[int(element_id)]
    material = model.get_material(element.material_name)
    if dof_mapping is None:
        dof_mapping = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp)
    else:
        dof_mapping = np.asarray(dof_mapping, dtype=np.intp)
    if (
        dof_mapping.size == 0
        or int(dof_mapping.min()) < 0
        or int(dof_mapping.max()) >= displacements.size
    ):
        if bool(getattr(element, "recovery_errors_fail_closed", False)):
            raise ValueError(
                "fail-closed element recovery requires a complete in-range "
                "DOF mapping"
            )
        return None
    try:
        return int(element_id), element.compute_stresses(
            model.mesh,
            displacements[dof_mapping],
            material,
            return_global=return_global,
        )
    except (IndexError, ValueError):
        if bool(getattr(element, "recovery_errors_fail_closed", False)):
            raise
        return None


def _compute_recovery_chunk(
    model: "FEModel",
    displacements: np.ndarray,
    items: Sequence[RecoveryPlanItem],
    *,
    return_global: bool,
) -> Tuple[Optional[Tuple[int, Dict[str, np.ndarray]]], ...]:
    return tuple(
        _compute_one_element_stress(
            model,
            displacements,
            item.element_id,
            return_global=return_global,
            dof_mapping=item.dof_mapping,
        )
        for item in items
    )


def _disabled_stress_recovery_result(
    resource_config: Optional[ResourceConfig],
) -> Tuple[Dict[int, Dict[str, np.ndarray]], RecoveryExecutionReport]:
    """Return the historical no-op result without observing mechanics."""

    report = RecoveryExecutionReport(
        phase="element_stress_recovery",
        item_count=0,
        requested_workers=(
            1
            if resource_config is None
            or resource_config.recovery_threads is None
            else int(resource_config.recovery_threads)
        ),
        used_workers=0,
        backend="disabled",
        deterministic=(
            True
            if resource_config is None
            else bool(resource_config.deterministic)
        ),
        elapsed_seconds=0.0,
        reason="stress recovery disabled",
    )
    return {}, report


def _recover_element_stresses_with_report_under_lease(
    model: "FEModel",
    displacements: np.ndarray,
    recovery_config: Optional[RecoveryConfig] = None,
    *,
    return_global: bool = False,
    resource_config: Optional[ResourceConfig] = None,
    qualified_runtime_guard: Any,
) -> Tuple[Dict[int, Dict[str, np.ndarray]], RecoveryExecutionReport]:
    """Recover element stresses and return bounded execution diagnostics."""

    qualified_runtime_guard(stage="configuration preflight")
    recovery = default_recovery_config(recovery_config)
    if not recovery.include_stresses:
        return _disabled_stress_recovery_result(resource_config)

    selected_ids = _ordered_element_ids(model, recovery.selected_element_ids(model))
    if return_global:
        require_model_element_capabilities(
            model,
            "global_recovery",
            context="recover_element_stresses_with_report",
            element_ids=selected_ids,
        )
    displacements = np.asarray(displacements, dtype=float)
    requested, used_workers, reason = _recovery_worker_count(resource_config, len(selected_ids))
    deterministic = True if resource_config is None else bool(resource_config.deterministic)
    backend = "serial" if used_workers <= 1 else "thread_pool"
    start = time.perf_counter()
    if len(selected_ids) < _MIN_RECOVERY_PLAN_SIZE:
        # Preserve the low-setup scalar oracle for small selections.  The
        # retained plan, native-pool scope and coarse scheduler break even only
        # on larger recoveries; paying them here regresses the mature path.
        stresses: Dict[int, Dict[str, np.ndarray]] = {}
        if used_workers <= 1:
            native_report: Dict[str, Any] = {
                "phase": "stress_recovery_serial",
                "status": "not_needed_serial",
                "requested_threads": None,
                "restored": True,
            }
            for element_id in selected_ids:
                item = _compute_one_element_stress(
                    model,
                    displacements,
                    element_id,
                    return_global=return_global,
                )
                if item is not None:
                    stresses[item[0]] = item[1]
        else:
            with native_thread_scope(
                1, phase="stress_recovery_thread_pool"
            ) as native_report:
                with ThreadPoolExecutor(max_workers=used_workers) as executor:
                    futures = [
                        executor.submit(
                            _compute_one_element_stress,
                            model,
                            displacements,
                            element_id,
                            return_global=return_global,
                        )
                        for element_id in selected_ids
                    ]
                    results = [future.result() for future in futures]
            for item in results:
                if item is not None:
                    stresses[item[0]] = item[1]
        components = recovery.selected_components()
        if components is not None:
            stresses = {
                int(element_id): _filter_components(values, components)
                for element_id, values in stresses.items()
            }
        report = RecoveryExecutionReport(
            phase="element_stress_recovery",
            item_count=len(selected_ids),
            requested_workers=requested,
            used_workers=used_workers,
            backend=backend,
            deterministic=deterministic,
            elapsed_seconds=float(time.perf_counter() - start),
            reason=reason,
            metadata={
                "recovery_backend": "scalar_legacy_small_selection",
                "batch_counts": {"scalar_legacy": len(selected_ids)},
                "compiled_batch_count": 0,
                "compiled_batch_seconds": 0.0,
                "eligible_element_count": 0,
                "fallback_element_count": len(selected_ids),
                "fallback_reasons": {
                    "below_recovery_plan_threshold": {
                        "element_ids": list(selected_ids),
                        "minimum_size": _MIN_RECOVERY_PLAN_SIZE,
                    }
                },
                "chunk_count": (
                    len(selected_ids) if used_workers > 1 else int(bool(selected_ids))
                ),
                "plan_reused": False,
                "plan_setup_seconds": 0.0,
                "plan_lookup_seconds": 0.0,
                "plan_retained_bytes": 0,
                "native_thread_policy": dict(native_report),
            },
        )
        return stresses, report

    plan_lookup_start = time.perf_counter()
    plan, plan_reused = _get_recovery_batch_plan_under_lease(
        model,
        qualified_runtime_guard=qualified_runtime_guard,
    )
    plan_lookup_seconds = time.perf_counter() - plan_lookup_start
    selected_items = plan.select(selected_ids)
    batch_counts = formulation_counts(selected_items)
    recovered_by_id: Dict[int, Dict[str, np.ndarray]] = {}
    compiled_ids: set[int] = set()
    reference_s3_ids: set[int] = set()
    compiled_seconds = 0.0
    reference_s3_seconds = 0.0
    compiled_error: Optional[str] = None
    reference_s3_error: Optional[str] = None
    candidate_indices = (
        np.empty(0, dtype=np.intp)
        if plan.isotropic_s4 is None
        else plan.isotropic_s4.select_indices(selected_ids)
    )
    compiled_active = bool(
        JIT_ENABLED
        and plan.isotropic_s4 is not None
        and candidate_indices.size >= _MIN_COMPILED_RECOVERY_BATCH
    )
    if compiled_active:
        compiled_start = time.perf_counter()
        try:
            with numba_thread_scope(used_workers):
                recovered_by_id.update(
                    recover_isotropic_s4(
                        plan.isotropic_s4,
                        candidate_indices,
                        displacements,
                        return_global=return_global,
                    )
                )
            compiled_ids.update(recovered_by_id)
        except Exception as exc:  # scalar oracle is the production fallback
            recovered_by_id.clear()
            compiled_ids.clear()
            compiled_error = f"{type(exc).__name__}: {exc}"
        compiled_seconds = time.perf_counter() - compiled_start

    reference_s3_indices = (
        np.empty(0, dtype=np.intp)
        if plan.reference_s3 is None
        else plan.reference_s3.select_indices(selected_ids)
    )
    reference_s3_active = bool(
        plan.reference_s3 is not None
        and reference_s3_indices.size >= MIN_REFERENCE_S3_RECOVERY_GROUP
    )
    if reference_s3_active:
        reference_s3_start = time.perf_counter()
        try:
            recovered_s3 = recover_reference_s3(
                model,
                plan.reference_s3,
                reference_s3_indices,
                displacements,
                return_global=return_global,
            )
            recovered_by_id.update(recovered_s3)
            reference_s3_ids.update(int(element_id) for element_id in recovered_s3)
        except Exception as exc:  # formulation-native scalar path remains authoritative
            reference_s3_error = f"{type(exc).__name__}: {exc}"
        reference_s3_seconds = time.perf_counter() - reference_s3_start

    fallback_items = tuple(
        item
        for item in selected_items
        if item.element_id not in compiled_ids
        and item.element_id not in reference_s3_ids
    )
    chunks = build_recovery_chunks(fallback_items, used_workers)
    if not fallback_items:
        native_report: Mapping[str, Any] = {
            "status": "not_needed",
            "requested_threads": None,
            "restored": True,
        }
    elif used_workers <= 1:
        for plan_item in fallback_items:
            item = _compute_one_element_stress(
                model,
                displacements,
                plan_item.element_id,
                return_global=return_global,
                dof_mapping=plan_item.dof_mapping,
            )
            if item is not None:
                recovered_by_id[item[0]] = item[1]
        native_report = {
            "status": "not_needed",
            "requested_threads": None,
            "restored": True,
        }
    else:
        with native_thread_scope(1, phase="stress_recovery_thread_pool") as native_report:
            with ThreadPoolExecutor(max_workers=used_workers) as executor:
                futures = [
                    executor.submit(
                        _compute_recovery_chunk,
                        model,
                        displacements,
                        chunk,
                        return_global=return_global,
                    )
                    for chunk in chunks
                ]
                chunk_results = [future.result() for future in futures]
        recovered_by_id.update({
            item[0]: item[1]
            for results in chunk_results
            for item in results
            if item is not None
        })

    stresses = {
        int(element_id): recovered_by_id[int(element_id)]
        for element_id in selected_ids
        if int(element_id) in recovered_by_id
    }

    components = recovery.selected_components()
    if components is not None:
        stresses = {int(element_id): _filter_components(values, components) for element_id, values in stresses.items()}
    elapsed = time.perf_counter() - start
    if compiled_ids and reference_s3_ids and fallback_items:
        recovery_backend = "hybrid_numba_qualified_s3_scalar"
    elif compiled_ids and reference_s3_ids:
        recovery_backend = "compiled_isotropic_s4_and_qualified_s3_reference"
    elif reference_s3_ids and fallback_items:
        recovery_backend = (
            "hybrid_qualified_s3_reference_thread_pool"
            if used_workers > 1
            else "hybrid_qualified_s3_reference_serial"
        )
    elif reference_s3_ids:
        recovery_backend = "qualified_s3_reference_elastic_shared_components"
    elif compiled_ids and fallback_items:
        recovery_backend = (
            "hybrid_numba_thread_pool" if used_workers > 1 else "hybrid_numba_serial"
        )
    elif compiled_ids:
        recovery_backend = "compiled_isotropic_s4"
    else:
        recovery_backend = (
            "scalar_serial" if used_workers <= 1 else "scalar_chunk_thread_pool"
        )
    # Preserve the public report contract.  Detailed backend selection is
    # exposed through metadata so existing callers that distinguish only
    # serial/threaded execution continue to work unchanged.
    backend = "serial" if used_workers <= 1 else "thread_pool"

    fallback_reasons: Dict[str, Any] = {}
    reference_s3_index = (
        {}
        if plan.reference_s3 is None
        else plan.reference_s3.index_by_id
    )
    recorded_reference_s3_candidates = set(plan.reference_s3_candidate_ids)
    unsupported_ids = [
        int(item.element_id)
        for item in fallback_items
        if (
            (plan.isotropic_s4 is None
            or int(item.element_id) not in plan.isotropic_s4.index_by_id)
            and int(item.element_id) not in reference_s3_index
            and int(item.element_id) not in recorded_reference_s3_candidates
        )
    ]
    eligible_fallback_ids = [
        int(item.element_id)
        for item in fallback_items
        if plan.isotropic_s4 is not None
        and int(item.element_id) in plan.isotropic_s4.index_by_id
    ]
    if unsupported_ids:
        fallback_reasons["unsupported_formulation"] = unsupported_ids
    if eligible_fallback_ids:
        if compiled_error is not None:
            fallback_reasons["compiled_batch_error"] = {
                "element_ids": eligible_fallback_ids,
                "error": compiled_error,
            }
        elif not JIT_ENABLED:
            fallback_reasons["jit_disabled"] = {
                "element_ids": eligible_fallback_ids,
                "reason": JIT_DISABLED_REASON,
            }
        else:
            fallback_reasons["batch_below_minimum_size"] = {
                "element_ids": eligible_fallback_ids,
                "minimum_size": _MIN_COMPILED_RECOVERY_BATCH,
            }
    eligible_reference_s3_fallback_ids = [
        int(item.element_id)
        for item in fallback_items
        if int(item.element_id) in reference_s3_index
    ]
    if eligible_reference_s3_fallback_ids:
        if reference_s3_error is not None:
            fallback_reasons["qualified_s3_reference_batch_error"] = {
                "element_ids": eligible_reference_s3_fallback_ids,
                "error": reference_s3_error,
            }
        else:
            fallback_reasons["qualified_s3_batch_below_minimum_size"] = {
                "element_ids": eligible_reference_s3_fallback_ids,
                "minimum_size": MIN_REFERENCE_S3_RECOVERY_GROUP,
            }
    for reason_name, reason_ids in plan.reference_s3_fallback_reasons.items():
        selected_reason_ids = [
            int(item.element_id)
            for item in fallback_items
            if int(item.element_id) in set(reason_ids)
        ]
        if selected_reason_ids:
            fallback_reasons[f"qualified_s3_{reason_name}"] = selected_reason_ids

    report = RecoveryExecutionReport(
        phase="element_stress_recovery",
        item_count=len(selected_ids),
        requested_workers=requested,
        used_workers=used_workers,
        backend=backend,
        deterministic=deterministic,
        elapsed_seconds=float(elapsed),
        reason=reason,
        metadata={
            "recovery_backend": recovery_backend,
            "batch_counts": batch_counts,
            "compiled_batch_count": int(bool(compiled_ids)),
            "compiled_batch_seconds": float(compiled_seconds),
            "qualified_s3_reference_batch_count": int(bool(reference_s3_ids)),
            "qualified_s3_reference_batch_seconds": float(reference_s3_seconds),
            "qualified_s3_reference_batch": (
                None
                if plan.reference_s3 is None
                else plan.reference_s3.diagnostics()
            ),
            "eligible_element_count": len(compiled_ids | reference_s3_ids),
            "fallback_element_count": len(fallback_items),
            "fallback_reasons": fallback_reasons,
            "chunk_count": len(chunks) if used_workers > 1 else int(bool(fallback_items)),
            "plan_reused": bool(plan_reused),
            "plan_setup_seconds": float(plan.setup_seconds),
            "plan_lookup_seconds": float(plan_lookup_seconds),
            "plan_retained_bytes": int(plan.retained_bytes),
            "native_thread_policy": dict(native_report),
        },
    )
    return stresses, report


def recover_element_stresses_with_report(
    model: "FEModel",
    displacements: np.ndarray,
    recovery_config: Optional[RecoveryConfig] = None,
    *,
    return_global: bool = False,
    resource_config: Optional[ResourceConfig] = None,
) -> Tuple[Dict[int, Dict[str, np.ndarray]], RecoveryExecutionReport]:
    """Recover stresses under one complete-operation authority lease."""

    recovery = default_recovery_config(recovery_config)
    if not recovery.include_stresses:
        # Disabled stress recovery is a true no-op.  Preserve that contract
        # before inspecting qualified element mechanics or their authority;
        # enabled recovery still enters the complete non-renewable lease.
        return _disabled_stress_recovery_result(resource_config)
    return _run_with_qualified_recovery_runtime_lease(
        model,
        context="element stress recovery",
        operation=lambda guard: _recover_element_stresses_with_report_under_lease(
            model,
            displacements,
            recovery,
            return_global=return_global,
            resource_config=resource_config,
            qualified_runtime_guard=guard,
        ),
    )


def recover_element_stresses(
    model: "FEModel",
    displacements: np.ndarray,
    recovery_config: Optional[RecoveryConfig] = None,
    *,
    return_global: bool = False,
    resource_config: Optional[ResourceConfig] = None,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Recover selected element stresses with optional component filtering."""

    stresses, _report = recover_element_stresses_with_report(
        model,
        displacements,
        recovery_config,
        return_global=return_global,
        resource_config=resource_config,
    )
    return stresses


_SHELL_TENSOR_COMPONENTS = ("xx", "yy", "zz", "xy", "yz", "xz")
_SHELL_SURFACE_KEYS = tuple(
    f"global_{component}_{surface}"
    for surface in ("top", "bot")
    for component in _SHELL_TENSOR_COMPONENTS
)


def _plane_stress_matrix(elastic_modulus: float, poisson_ratio: float) -> np.ndarray:
    nu = float(poisson_ratio)
    return float(elastic_modulus) / max(1.0 - nu**2, 1.0e-12) * np.array(
        [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]],
        dtype=float,
    )


def _rotate_shell_surface_tensors(
    stresses: Mapping[str, Any],
    rotation: np.ndarray,
) -> Dict[str, Any]:
    """Rotate already-global shell surface tensors by a rigid rotation."""

    rotated: Dict[str, Any] = dict(stresses)
    R = np.asarray(rotation, dtype=float).reshape(3, 3)
    first_key = "global_xx_top"
    if first_key not in stresses:
        return rotated
    count = np.asarray(stresses[first_key], dtype=float).reshape(-1).size
    for surface in ("top", "bot"):
        output = {
            component: np.zeros(count, dtype=float)
            for component in _SHELL_TENSOR_COMPONENTS
        }
        for index in range(count):
            tensor = np.array(
                [
                    [
                        np.asarray(stresses[f"global_xx_{surface}"], dtype=float).reshape(-1)[index],
                        np.asarray(stresses[f"global_xy_{surface}"], dtype=float).reshape(-1)[index],
                        np.asarray(stresses[f"global_xz_{surface}"], dtype=float).reshape(-1)[index],
                    ],
                    [
                        np.asarray(stresses[f"global_xy_{surface}"], dtype=float).reshape(-1)[index],
                        np.asarray(stresses[f"global_yy_{surface}"], dtype=float).reshape(-1)[index],
                        np.asarray(stresses[f"global_yz_{surface}"], dtype=float).reshape(-1)[index],
                    ],
                    [
                        np.asarray(stresses[f"global_xz_{surface}"], dtype=float).reshape(-1)[index],
                        np.asarray(stresses[f"global_yz_{surface}"], dtype=float).reshape(-1)[index],
                        np.asarray(stresses[f"global_zz_{surface}"], dtype=float).reshape(-1)[index],
                    ],
                ],
                dtype=float,
            )
            tensor = R @ tensor @ R.T
            output["xx"][index] = tensor[0, 0]
            output["yy"][index] = tensor[1, 1]
            output["zz"][index] = tensor[2, 2]
            output["xy"][index] = tensor[0, 1]
            output["yz"][index] = tensor[1, 2]
            output["xz"][index] = tensor[0, 2]
        for component, values in output.items():
            rotated[f"global_{component}_{surface}"] = values
    return rotated


def _element_recovery_context(
    model: "FEModel",
    element_id: int,
    element: Any,
    displacements: np.ndarray,
    *,
    kinematics: str,
    return_global: bool,
    stable_shell_rotation: bool = False,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Elastic component recovery and frame using the solved kinematics."""

    material = model.get_material(element.material_name)
    reference_coords = np.asarray(element.get_node_coordinates(model.mesh), dtype=float)
    stress_frame = (
        np.asarray(element._center_frame(reference_coords), dtype=float)
        if hasattr(element, "_center_frame")
        else np.eye(3, dtype=float)
    )
    recovery_coords = reference_coords
    if str(kinematics) != "corotational":
        values = element.compute_stresses(
            model.mesh,
            displacements,
            material,
            return_global=return_global,
        )
        return dict(values), stress_frame, recovery_coords

    from .corotational import (
        _corotational_cache,
        _corotational_deformation_state,
    )

    reference = _corotational_cache(model).get(int(element_id))
    if reference is None:
        values = element.compute_stresses(
            model.mesh,
            displacements,
            material,
            return_global=return_global,
        )
        return dict(values), stress_frame, recovery_coords
    dof_mapping = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp)
    u_deformational, rigid_rotation, recovery_coords = (
        _corotational_deformation_state(
            reference,
            element,
            np.asarray(displacements, dtype=float)[dof_mapping],
            stable_shell_rotation=stable_shell_rotation,
        )
    )
    element_displacements = np.zeros_like(np.asarray(displacements, dtype=float))
    element_displacements[dof_mapping] = u_deformational
    values = dict(
        element.compute_stresses(
            model.mesh,
            element_displacements,
            material,
            return_global=return_global,
        )
    )
    if return_global:
        values = _rotate_shell_surface_tensors(values, rigid_rotation)
    if getattr(reference, "category", None) == "shell":
        stress_frame = rigid_rotation @ np.asarray(reference.frame, dtype=float)
    return values, stress_frame, np.asarray(recovery_coords, dtype=float)


def _state_uses_plastic_constitutive_history(
    material: Any,
    state: Mapping[str, Any],
) -> bool:
    """Return whether a committed state belongs to an active plastic model."""

    if getattr(material, "hardening_curve", None) is not None:
        return True
    for key in ("alpha", "plastic_strain"):
        values = np.asarray(state.get(key, ()), dtype=float).reshape(-1)
        if values.size and np.any(np.abs(values) > np.finfo(float).eps):
            return True
    return False


def _hill_current_x_strength(material: Any, alpha: np.ndarray) -> np.ndarray:
    """Directional-X strength at committed Hill equivalent plastic strain."""

    hill_yield = getattr(material, "hill_yield", None)
    if hill_yield is None:
        raise ValueError("Hill current strength requires material.hill_yield")
    alpha_values = np.asarray(alpha, dtype=float)
    curve = getattr(material, "hardening_curve", None)
    if curve is None:
        return np.full(alpha_values.shape, float(hill_yield.X), dtype=float)
    reference = float(
        np.asarray(curve.flow_stress(np.array([0.0])), dtype=float).reshape(-1)[0]
    )
    if not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("Hill hardening curve must have positive flow_stress(0)")
    return (
        float(hill_yield.X)
        * np.asarray(curve.flow_stress(alpha_values), dtype=float)
        / reference
    )


def _element_has_generalized_section(element: Any) -> bool:
    """Return whether an element owns a pre-integrated section contract."""

    return (
        getattr(element, "shell_section", None) is not None
        or getattr(element, "generalized_section", None) is not None
    )


def _committed_generalized_shell_state(
    model: "FEModel",
    element_id: int,
    element: Any,
    state: Any,
    *,
    displacements: np.ndarray,
    kinematics: str,
    return_global: bool,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Recover a generalized shell state without reconstructing ply stress.

    The nonlinear element has already evaluated the von Karman strain and the
    A/B/D/As resultants at its constitutive integration stations.  Re-evaluating
    ``compute_stresses`` would silently replace that state with a linear
    displacement reconstruction, so the committed arrays are the sole source
    of these resultants.
    """

    from .elements import ShellElement
    from .shell_sections import (
        SHELL_MEMBRANE_VOIGT_ORDER,
        SHELL_TRANSVERSE_SHEAR_ORDER,
    )

    if not isinstance(element, ShellElement) or element.shell_section is None:
        return None, "element has no generalized shell section"
    if not isinstance(state, Mapping):
        return None, "committed generalized shell state is not a mapping"

    def matrix(name: str, width: int) -> Optional[np.ndarray]:
        values = np.asarray(state.get(name, ()), dtype=float)
        if values.size == 0:
            return None
        if values.ndim == 1 and values.shape == (width,):
            values = values.reshape(1, width)
        if values.ndim != 2 or values.shape[1] != width:
            return None
        if not np.all(np.isfinite(values)):
            return None
        return values.copy()

    membrane_strain = matrix("membrane_strain", 3)
    curvature = matrix("curvature", 3)
    membrane_resultants = matrix("membrane_resultants", 3)
    bending_resultants = matrix("bending_resultants", 3)
    transverse_shear_strain = matrix("transverse_shear_strain", 2)
    transverse_shear_resultants = matrix(
        "transverse_shear_resultants",
        2,
    )
    if any(
        values is None
        for values in (
            membrane_strain,
            curvature,
            membrane_resultants,
            bending_resultants,
            transverse_shear_strain,
            transverse_shear_resultants,
        )
    ):
        return None, (
            "committed generalized shell state lacks finite generalized "
            "strain/resultant arrays"
        )
    assert membrane_strain is not None
    assert curvature is not None
    assert membrane_resultants is not None
    assert bending_resultants is not None
    assert transverse_shear_strain is not None
    assert transverse_shear_resultants is not None
    if not (
        membrane_strain.shape
        == curvature.shape
        == membrane_resultants.shape
        == bending_resultants.shape
    ):
        return None, (
            "committed generalized shell membrane/bending arrays have "
            "incompatible shapes"
        )
    if transverse_shear_strain.shape != transverse_shear_resultants.shape:
        return None, (
            "committed generalized shell transverse-shear arrays have "
            "incompatible shapes"
        )
    membrane_order = tuple(
        state.get("membrane_resultant_order", SHELL_MEMBRANE_VOIGT_ORDER)
    )
    shear_order = tuple(
        state.get(
            "transverse_shear_resultant_order",
            SHELL_TRANSVERSE_SHEAR_ORDER,
        )
    )
    if membrane_order != tuple(SHELL_MEMBRANE_VOIGT_ORDER):
        return None, "committed generalized shell membrane order is unsupported"
    if shear_order != tuple(SHELL_TRANSVERSE_SHEAR_ORDER):
        return None, (
            "committed generalized shell transverse-shear order is unsupported"
        )

    recovered: Dict[str, Any] = {
        "generalized_stress_scope": "section_resultants_only",
        "recovery_scope": "section_resultants_only",
        "physical_stress_available": False,
        "membrane_resultant_order": SHELL_MEMBRANE_VOIGT_ORDER,
        "transverse_shear_resultant_order": SHELL_TRANSVERSE_SHEAR_ORDER,
        "membrane_strain": membrane_strain,
        "curvature": curvature,
        "transverse_shear_strain": transverse_shear_strain,
        "membrane_resultants": membrane_resultants,
        "bending_resultants": bending_resultants,
        "transverse_shear_resultants": transverse_shear_resultants,
    }
    if return_global:
        reference_coordinates = np.asarray(
            element.get_node_coordinates(model.mesh),
            dtype=float,
        )
        recovery_coordinates = reference_coordinates
        stress_frame = (
            np.asarray(element._center_frame(reference_coordinates), dtype=float)
            if hasattr(element, "_center_frame")
            else np.eye(3, dtype=float)
        )
        if str(kinematics) == "corotational":
            from .corotational import (
                _corotational_cache,
                _corotational_deformation_state,
            )

            reference = _corotational_cache(model).get(int(element_id))
            if reference is not None:
                dof_mapping = np.asarray(
                    element.get_dof_mapping(model.mesh),
                    dtype=np.intp,
                )
                _u_deformational, rigid_rotation, recovery_coordinates = (
                    _corotational_deformation_state(
                        reference,
                        element,
                        np.asarray(displacements, dtype=float)[dof_mapping],
                    )
                )
                if getattr(reference, "category", None) == "shell":
                    stress_frame = (
                        rigid_rotation
                        @ np.asarray(reference.frame, dtype=float)
                    )

        def rotate_tensor_rows(values: np.ndarray) -> np.ndarray:
            output = np.zeros((values.shape[0], 3, 3), dtype=float)
            for index, row in enumerate(values):
                local = np.array(
                    [
                        [row[0], row[2], 0.0],
                        [row[2], row[1], 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                    dtype=float,
                )
                output[index] = stress_frame @ local @ stress_frame.T
            return output

        recovered.update(
            {
                "global_membrane_resultant_tensors": rotate_tensor_rows(
                    membrane_resultants
                ),
                "global_bending_resultant_tensors": rotate_tensor_rows(
                    bending_resultants
                ),
                "global_transverse_shear_resultants": (
                    transverse_shear_resultants[:, 0, None]
                    * stress_frame[:, 0][None, :]
                    + transverse_shear_resultants[:, 1, None]
                    * stress_frame[:, 1][None, :]
                ),
                "_recovery_coordinates": np.asarray(
                    recovery_coordinates,
                    dtype=float,
                ).copy(),
                "_recovery_stress_frame": stress_frame.copy(),
            }
        )
    return recovered, ""


def _committed_generalized_beam_state(
    element: Any,
    state: Any,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Recover exact committed generalized beam strains and resultants."""

    from .elements import (
        BeamElement,
        GENERALIZED_BEAM_RESULTANT_ORDER,
        GENERALIZED_BEAM_STRAIN_ORDER,
        QuadraticBeamElement,
        _generalized_beam_recovery,
    )

    if not isinstance(element, (BeamElement, QuadraticBeamElement)):
        return None, "element is not a beam"
    if getattr(element, "generalized_section", None) is None:
        return None, "element has no generalized beam section"
    if not isinstance(state, Mapping):
        return None, "committed generalized beam state is not a mapping"
    strains = np.asarray(state.get("generalized_strain", ()), dtype=float)
    resultants = np.asarray(
        state.get("generalized_resultant", ()),
        dtype=float,
    )
    if strains.ndim == 1 and strains.shape == (6,):
        strains = strains.reshape(1, 6)
    if resultants.ndim == 1 and resultants.shape == (6,):
        resultants = resultants.reshape(1, 6)
    if (
        strains.ndim != 2
        or strains.shape[1] != 6
        or resultants.shape != strains.shape
        or not np.all(np.isfinite(strains))
        or not np.all(np.isfinite(resultants))
    ):
        return None, (
            "committed generalized beam state lacks compatible finite "
            "generalized_strain/generalized_resultant arrays"
        )
    strain_order = tuple(
        state.get("generalized_strain_order", GENERALIZED_BEAM_STRAIN_ORDER)
    )
    resultant_order = tuple(
        state.get(
            "generalized_resultant_order",
            GENERALIZED_BEAM_RESULTANT_ORDER,
        )
    )
    if strain_order != tuple(GENERALIZED_BEAM_STRAIN_ORDER):
        return None, "committed generalized beam strain order is unsupported"
    if resultant_order != tuple(GENERALIZED_BEAM_RESULTANT_ORDER):
        return None, "committed generalized beam resultant order is unsupported"

    stations = np.asarray(
        state.get("generalized_station_coordinates", ()),
        dtype=float,
    ).reshape(-1)
    if stations.size == 0:
        stations = (
            np.asarray(element.GAUSS_POINTS, dtype=float)
            if isinstance(element, QuadraticBeamElement)
            else np.array((-1.0, 0.0, 1.0), dtype=float)
        )
    if (
        stations.shape != (strains.shape[0],)
        or not np.all(np.isfinite(stations))
    ):
        return None, (
            "committed generalized beam station coordinates are incompatible"
        )
    return (
        _generalized_beam_recovery(
            strains.copy(),
            resultants.copy(),
            stations=stations.copy(),
        ),
        "",
    )


def _recover_generalized_committed_state(
    model: "FEModel",
    element_id: int,
    state: Any,
    *,
    displacements: np.ndarray,
    kinematics: str,
    return_global: bool,
) -> Tuple[Optional[Dict[str, Any]], str, str, Dict[str, str]]:
    """Dispatch exact resultants-only recovery for a generalized section."""

    element = model.mesh.elements.get(int(element_id))
    if element is None:
        return None, "", "committed state references an unknown element", {}
    if getattr(element, "shell_section", None) is not None:
        recovered, reason = _committed_generalized_shell_state(
            model,
            int(element_id),
            element,
            state,
            displacements=displacements,
            kinematics=kinematics,
            return_global=return_global,
        )
        if recovered is None:
            return None, "", reason, {}
        return (
            recovered,
            "committed_generalized_shell_section_state",
            "",
            {
                "generalized_membrane_bending": (
                    "committed_generalized_shell_section_state"
                ),
                "generalized_transverse_shear": (
                    "committed_generalized_shell_section_state"
                ),
                "physical_stress": "unavailable_from_preintegrated_section",
                "stress_frame": (
                    "current_corotated_center_frame"
                    if str(kinematics) == "corotational"
                    else "reference_center_frame"
                ),
            },
        )
    if getattr(element, "generalized_section", None) is not None:
        recovered, reason = _committed_generalized_beam_state(element, state)
        if recovered is None:
            return None, "", reason, {}
        return (
            recovered,
            "committed_generalized_beam_section_state",
            "",
            {
                "generalized_strain_and_resultant": (
                    "committed_generalized_beam_section_state"
                ),
                "physical_stress": "unavailable_from_generalized_section",
                "stress_frame": (
                    "current_corotated_frame"
                    if str(kinematics) == "corotational"
                    else "reference_frame"
                ),
            },
        )
    return None, "", "element has no generalized section", {}


def _state_shell_stresses(
    model: "FEModel",
    element: Any,
    state: Any,
    elastic_stresses: Mapping[str, Any],
    *,
    return_global: bool,
    stress_frame: np.ndarray,
    recovery_coordinates: np.ndarray,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Recover one shell from its committed layer state."""

    from .elements import ShellElement
    from .plasticity import lobatto_layers

    if not isinstance(element, ShellElement):
        return None, "element is not a shell"
    if not isinstance(state, Mapping):
        return None, "committed state is not a mapping"

    n_gp = int(len(element.gauss_points))
    layer_strain = np.asarray(state.get("layer_strain", ()), dtype=float)
    if layer_strain.size == 0 or layer_strain.size % 3:
        return None, "committed shell state has no valid layer_strain"
    try:
        layer_strain = layer_strain.reshape(-1, 3)
    except ValueError:
        return None, "committed shell layer_strain has an invalid shape"
    if n_gp <= 0 or layer_strain.shape[0] % n_gp:
        return None, "committed shell layer count is incompatible with its Gauss rule"
    num_layers = layer_strain.shape[0] // n_gp

    material = model.get_material(element.material_name)
    layer_stress = np.asarray(state.get("layer_stress", ()), dtype=float)
    if layer_stress.size:
        if layer_stress.size != layer_strain.size:
            return None, "committed shell layer_stress shape does not match layer_strain"
        layer_stress = layer_stress.reshape(layer_strain.shape)
    else:
        plastic_strain = np.asarray(state.get("plastic_strain", ()), dtype=float)
        if plastic_strain.size != layer_strain.size:
            return None, "committed shell state lacks compatible layer_stress/plastic_strain"
        from .materials import is_isotropic_material

        if not is_isotropic_material(material):
            return None, (
                "orthotropic committed shell state lacks stored physical layer_stress"
            )
        layer_stress = (layer_strain - plastic_strain.reshape(layer_strain.shape)) @ _plane_stress_matrix(
            material.elastic_modulus,
            material.poisson_ratio,
        ).T
    if not np.all(np.isfinite(layer_strain)) or not np.all(np.isfinite(layer_stress)):
        return None, "committed shell layer state contains non-finite values"

    try:
        z_layers, layer_weights = lobatto_layers(num_layers, float(element.thickness))
    except ValueError:
        return None, f"unsupported committed shell layer count {num_layers}"
    stress_layers = layer_stress.reshape(n_gp, num_layers, 3)
    strain_layers = layer_strain.reshape(n_gp, num_layers, 3)
    thickness = max(abs(float(element.thickness)), np.finfo(float).tiny)
    membrane_resultants = np.einsum(
        "l,gli->gi", layer_weights, stress_layers
    )
    bending_resultants = np.einsum(
        "l,l,gli->gi", layer_weights, z_layers, stress_layers
    )
    membrane_stress = membrane_resultants / thickness
    bending_stress = (
        6.0
        * bending_resultants
        / thickness**2
    )
    membrane_strain = np.einsum("l,gli->gi", layer_weights, strain_layers) / thickness

    shear_xz = np.asarray(elastic_stresses.get("shear_xz", np.zeros(n_gp)), dtype=float).reshape(-1)
    shear_yz = np.asarray(elastic_stresses.get("shear_yz", np.zeros(n_gp)), dtype=float).reshape(-1)
    if shear_xz.size != n_gp:
        shear_xz = np.zeros(n_gp, dtype=float)
    if shear_yz.size != n_gp:
        shear_yz = np.zeros(n_gp, dtype=float)

    sx = stress_layers[:, :, 0]
    sy = stress_layers[:, :, 1]
    txy = stress_layers[:, :, 2]
    # The layered shell constitutive update is plane-stress J2.  Transverse
    # shear is reconstructed elastically from the matching displacement field,
    # but it is not part of the return-mapped material state.  Keep the public
    # ``von_mises`` field history-consistent and expose the mixed diagnostic
    # separately; otherwise large elastic transverse shear can falsely appear
    # to violate the selected hardening curve.
    vm_layers = np.sqrt(
        np.maximum(
            sx**2
            - sx * sy
            + sy**2
            + 3.0 * txy**2,
            0.0,
        )
    )
    mixed_vm_layers = np.sqrt(
        np.maximum(
            vm_layers**2
            + 3.0
            * (
                shear_xz[:, None] ** 2
                + shear_yz[:, None] ** 2
            ),
            0.0,
        )
    )
    material_history_active = _state_uses_plastic_constitutive_history(
        material,
        state,
    )
    from .materials import is_orthotropic_material

    primary_vm_layers = (
        mixed_vm_layers
        if is_orthotropic_material(material)
        else (vm_layers if material_history_active else mixed_vm_layers)
    )
    recovered: Dict[str, Any] = dict(elastic_stresses)
    recovered.update(
        {
            "membrane_strain_xx": membrane_strain[:, 0].copy(),
            "membrane_strain_yy": membrane_strain[:, 1].copy(),
            "membrane_strain_xy": membrane_strain[:, 2].copy(),
            "membrane_resultants": membrane_resultants.copy(),
            "bending_resultants": bending_resultants.copy(),
            "membrane_xx": membrane_stress[:, 0].copy(),
            "membrane_yy": membrane_stress[:, 1].copy(),
            "membrane_xy": membrane_stress[:, 2].copy(),
            "bending_xx": bending_stress[:, 0].copy(),
            "bending_yy": bending_stress[:, 1].copy(),
            "bending_xy": bending_stress[:, 2].copy(),
            "shear_xz": shear_xz.copy(),
            "shear_yz": shear_yz.copy(),
            "von_mises": np.max(primary_vm_layers, axis=1),
            "in_plane_von_mises": np.max(vm_layers, axis=1),
            "mixed_reconstruction_von_mises": np.max(
                mixed_vm_layers, axis=1
            ),
        }
    )
    hill_yield = getattr(material, "hill_yield", None)
    material_layer_stress = np.asarray(
        state.get("layer_stress_material", ()),
        dtype=float,
    )
    if hill_yield is not None:
        if material_layer_stress.size != layer_stress.size:
            return None, (
                "orthotropic Hill shell state lacks compatible "
                "material-axis layer_stress_material"
            )
        from .plasticity import hill48_plane_stress_equivalent_stress

        material_layer_stress = material_layer_stress.reshape(layer_stress.shape)
        hill_equivalent = hill48_plane_stress_equivalent_stress(
            material_layer_stress,
            hill_yield,
        ).reshape(n_gp, num_layers)
        alpha = np.asarray(state.get("alpha", ()), dtype=float).reshape(-1)
        if alpha.size == 0:
            alpha = np.zeros(layer_stress.shape[0], dtype=float)
        elif alpha.size == 1:
            alpha = np.full(layer_stress.shape[0], float(alpha[0]), dtype=float)
        elif alpha.size != layer_stress.shape[0]:
            return None, "orthotropic committed shell alpha shape is incompatible"
        current_strength = _hill_current_x_strength(material, alpha).reshape(
            n_gp,
            num_layers,
        )
        recovered["equivalent_stress"] = np.max(hill_equivalent, axis=1)
        recovered["equivalent_stress_measure"] = "hill48"
        recovered["hill_utilization"] = np.max(
            hill_equivalent
            / np.maximum(current_strength, np.finfo(float).tiny),
            axis=1,
        )
        recovered["equivalent_stress_scope"] = (
            "return-mapped material-axis plane-stress layers"
        )
    else:
        recovered["equivalent_stress"] = recovered["von_mises"].copy()
        recovered["equivalent_stress_measure"] = "von_mises"

    if return_global:
        local_surface: Dict[str, np.ndarray] = {
            f"local_{component}_{surface}": np.zeros(n_gp, dtype=float)
            for surface in ("top", "bot")
            for component in _SHELL_TENSOR_COMPONENTS
        }
        global_surface: Dict[str, np.ndarray] = {
            key: np.zeros(n_gp, dtype=float) for key in _SHELL_SURFACE_KEYS
        }
        rotation = np.asarray(stress_frame, dtype=float).reshape(3, 3)
        for gp_index, (_xi, _eta) in enumerate(np.asarray(element.gauss_points, dtype=float)):
            for surface, layer_index in (("bot", 0), ("top", -1)):
                shell_stress = stress_layers[gp_index, layer_index]
                local_tensor = np.array(
                    [
                        [shell_stress[0], shell_stress[2], shear_xz[gp_index]],
                        [shell_stress[2], shell_stress[1], shear_yz[gp_index]],
                        [shear_xz[gp_index], shear_yz[gp_index], 0.0],
                    ],
                    dtype=float,
                )
                global_tensor = rotation @ local_tensor @ rotation.T
                local_values = {
                    "xx": local_tensor[0, 0],
                    "yy": local_tensor[1, 1],
                    "zz": local_tensor[2, 2],
                    "xy": local_tensor[0, 1],
                    "yz": local_tensor[1, 2],
                    "xz": local_tensor[0, 2],
                }
                global_values = {
                    "xx": global_tensor[0, 0],
                    "yy": global_tensor[1, 1],
                    "zz": global_tensor[2, 2],
                    "xy": global_tensor[0, 1],
                    "yz": global_tensor[1, 2],
                    "xz": global_tensor[0, 2],
                }
                for component in _SHELL_TENSOR_COMPONENTS:
                    local_surface[f"local_{component}_{surface}"][gp_index] = local_values[component]
                    global_surface[f"global_{component}_{surface}"][gp_index] = global_values[component]
        recovered.update(local_surface)
        recovered.update(global_surface)
    recovered["_recovery_coordinates"] = np.asarray(
        recovery_coordinates, dtype=float
    ).copy()
    recovered["_recovery_stress_frame"] = np.asarray(
        stress_frame, dtype=float
    ).copy()
    return recovered, ""


def _state_beam_stresses(
    model: "FEModel",
    element: Any,
    state: Any,
    elastic_stresses: Mapping[str, Any],
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Recover exact beam resultants from committed return-mapped fibers."""

    from .elements import BeamElement, QuadraticBeamElement

    if not isinstance(element, (BeamElement, QuadraticBeamElement)):
        return None, "element is not a beam"
    if not isinstance(state, Mapping):
        return None, "committed state is not a mapping"
    material = model.get_material(element.material_name)
    fiber_stress = np.asarray(state.get("fiber_stress", ()), dtype=float).reshape(-1)
    if fiber_stress.size == 0:
        return None, "committed beam state has no fiber_stress"
    if not np.all(np.isfinite(fiber_stress)):
        return None, "committed beam fiber_stress contains non-finite values"

    fiber_y = np.asarray(state.get("fiber_y", ()), dtype=float).reshape(-1)
    fiber_z = np.asarray(state.get("fiber_z", ()), dtype=float).reshape(-1)
    fiber_weights = np.asarray(state.get("fiber_weights", ()), dtype=float).reshape(-1)
    if not (
        fiber_y.size
        and fiber_y.size == fiber_z.size
        and fiber_y.size == fiber_weights.size
    ):
        config = element._fiber_plasticity_config(material)
        if config is None:
            return None, (
                "committed beam fiber state has no fiber layout and the element "
                "has no active fiber-plasticity configuration"
            )
        fiber_y, fiber_z, fiber_weights = (
            np.asarray(values, dtype=float).reshape(-1)
            for values in element._fiber_section_grid(config)
        )
    if (
        fiber_y.size == 0
        or np.any(~np.isfinite(fiber_y))
        or np.any(~np.isfinite(fiber_z))
        or np.any(~np.isfinite(fiber_weights))
        or np.any(fiber_weights <= 0.0)
    ):
        return None, "committed beam fiber layout is invalid"

    default_stations = (
        len(element.GAUSS_POINTS)
        if isinstance(element, QuadraticBeamElement)
        else 1
    )
    station_count = int(state.get("fiber_station_count", default_stations))
    expected_size = station_count * fiber_y.size
    if station_count <= 0 or fiber_stress.size != expected_size:
        return None, (
            "committed beam fiber_stress shape is incompatible with the "
            f"{station_count}x{fiber_y.size} station/fiber layout"
        )
    stress_by_station = fiber_stress.reshape(station_count, fiber_y.size)
    axial_force = stress_by_station @ fiber_weights
    moment_y = (stress_by_station * fiber_z[None, :]) @ fiber_weights
    moment_z = (stress_by_station * fiber_y[None, :]) @ fiber_weights
    area = max(float(np.sum(fiber_weights)), np.finfo(float).tiny)

    def signed_envelope(values: np.ndarray) -> float:
        flat = np.asarray(values, dtype=float).reshape(-1)
        return float(flat[int(np.argmax(np.abs(flat)))]) if flat.size else 0.0

    axial = float(np.mean(axial_force) / area)
    c_y, c_z = element._fiber_distances()
    bending_y_by_station = (
        moment_y * float(c_z) / max(float(element._Iy), np.finfo(float).tiny)
    )
    bending_z_by_station = (
        moment_z * float(c_y) / max(float(element._Iz), np.finfo(float).tiny)
    )
    shear_y = float(
        np.max(
            np.abs(
                np.asarray(
                    elastic_stresses.get("shear_stress_y", 0.0),
                    dtype=float,
                )
            )
        )
    )
    shear_z = float(
        np.max(
            np.abs(
                np.asarray(
                    elastic_stresses.get("shear_stress_z", 0.0),
                    dtype=float,
                )
            )
        )
    )
    torsion = float(
        np.max(
            np.abs(
                np.asarray(
                    elastic_stresses.get("torsional_stress", 0.0),
                    dtype=float,
                )
            )
        )
    )
    # Beam plasticity return-maps the axial fiber stress only.  Shear and
    # torsion are matching elastic reconstructions and therefore cannot be
    # folded into a material-history equivalent stress without implying a
    # three-dimensional return mapping that the element does not perform.
    fiber_von_mises = np.abs(stress_by_station)
    mixed_fiber_von_mises = np.sqrt(
        np.maximum(
            stress_by_station**2
            + 3.0 * (shear_y**2 + shear_z**2 + torsion**2),
            0.0,
        )
    )
    material_history_active = _state_uses_plastic_constitutive_history(
        material,
        state,
    )
    from .materials import is_orthotropic_material

    primary_fiber_von_mises = (
        mixed_fiber_von_mises
        if is_orthotropic_material(material)
        else (
            fiber_von_mises
            if material_history_active
            else mixed_fiber_von_mises
        )
    )
    recovered: Dict[str, Any] = dict(elastic_stresses)
    recovered.update(
        {
            "von_mises": float(np.max(primary_fiber_von_mises)),
            "axial_stress": axial,
            "bending_stress_y": signed_envelope(bending_y_by_station),
            "bending_stress_z": signed_envelope(bending_z_by_station),
            "shear_stress_y": shear_y,
            "shear_stress_z": shear_z,
            "torsional_stress": torsion,
            "longitudinal_fiber_von_mises": float(
                np.max(fiber_von_mises)
            ),
            "fiber_stress_min": float(np.min(stress_by_station)),
            "fiber_stress_max": float(np.max(stress_by_station)),
            "fiber_von_mises_max": float(np.max(fiber_von_mises)),
            "mixed_reconstruction_von_mises": float(
                np.max(mixed_fiber_von_mises)
            ),
            "fiber_mixed_reconstruction_von_mises_max": float(
                np.max(mixed_fiber_von_mises)
            ),
            "axial_force_by_station": axial_force.copy(),
            "bending_moment_y_by_station": moment_y.copy(),
            "bending_moment_z_by_station": moment_z.copy(),
            "fiber_station_count": station_count,
        }
    )
    hill_yield = getattr(material, "hill_yield", None)
    if hill_yield is not None:
        equivalent_by_fiber = np.abs(stress_by_station)
        alpha = np.asarray(state.get("alpha", ()), dtype=float).reshape(-1)
        if alpha.size == 0:
            alpha = np.zeros(stress_by_station.size, dtype=float)
        elif alpha.size == 1:
            alpha = np.full(stress_by_station.size, float(alpha[0]), dtype=float)
        elif alpha.size != stress_by_station.size:
            return None, "orthotropic committed beam alpha shape is incompatible"
        current_strength = _hill_current_x_strength(material, alpha).reshape(
            stress_by_station.shape
        )
        recovered["equivalent_stress"] = float(np.max(equivalent_by_fiber))
        recovered["equivalent_stress_measure"] = "hill48"
        recovered["hill_utilization"] = float(
            np.max(
                equivalent_by_fiber
                / np.maximum(current_strength, np.finfo(float).tiny)
            )
        )
        recovered["equivalent_stress_scope"] = (
            "committed longitudinal fibers; elastic shear/torsion excluded"
        )
    else:
        recovered["equivalent_stress"] = recovered["von_mises"]
        recovered["equivalent_stress_measure"] = "von_mises"
    return recovered, ""


def _recover_one_committed_state(
    model: "FEModel",
    element_id: int,
    state: Any,
    elastic_stresses: Mapping[str, Any],
    *,
    displacements: np.ndarray,
    kinematics: str,
    return_global: bool,
    stable_shell_rotation: bool = False,
) -> Tuple[Optional[Dict[str, Any]], str, str, Dict[str, str]]:
    element = model.mesh.elements.get(int(element_id))
    if element is None:
        return None, "", "committed state references an unknown element", {}
    if _is_qualified_s3(element):
        recovered, reason, component_sources = (
            _recover_qualified_s3_committed_state(
                model,
                int(element_id),
                element,
                state,
                displacements=displacements,
                return_global=return_global,
            )
        )
        return (
            recovered,
            (
                "committed_s3_v2d_hammer3_state"
                if recovered is not None and _is_v2d_s3(element)
                else "committed_qualified_s3_native_state"
                if recovered is not None
                else ""
            ),
            reason,
            component_sources,
        )
    if _element_has_generalized_section(element):
        return _recover_generalized_committed_state(
            model,
            int(element_id),
            state,
            displacements=displacements,
            kinematics=kinematics,
            return_global=return_global,
        )
    contextual_elastic, stress_frame, recovery_coordinates = (
        _element_recovery_context(
            model,
            int(element_id),
            element,
            displacements,
            kinematics=kinematics,
            return_global=False,
            stable_shell_rotation=stable_shell_rotation,
        )
        if str(kinematics) == "corotational"
        else (
            dict(elastic_stresses),
            (
                np.asarray(
                    element._center_frame(
                        element.get_node_coordinates(model.mesh)
                    ),
                    dtype=float,
                )
                if hasattr(element, "_center_frame")
                else np.eye(3, dtype=float)
            ),
            np.asarray(element.get_node_coordinates(model.mesh), dtype=float),
        )
    )
    shell, reason = _state_shell_stresses(
        model,
        element,
        state,
        contextual_elastic,
        return_global=return_global,
        stress_frame=stress_frame,
        recovery_coordinates=recovery_coordinates,
    )
    if shell is not None:
        shell_material = model.get_material(element.material_name)
        shell_material_history = _state_uses_plastic_constitutive_history(
            shell_material,
            state,
        )
        from .materials import is_orthotropic_material

        shell_is_orthotropic = is_orthotropic_material(shell_material)
        shell_uses_hill = (
            shell.get("equivalent_stress_measure") == "hill48"
        )
        return (
            shell,
            "committed_shell_layer_state",
            "",
            {
                "membrane_bending_and_surface_in_plane": "committed_shell_layer_state",
                "transverse_shear": "elastic_reconstruction_from_same_solution",
                "von_mises": (
                    "committed_shell_layer_state_plus_elastic_transverse_shear"
                    if shell_is_orthotropic
                    else (
                        "committed_shell_layer_state_in_plane"
                        if shell_material_history
                        else (
                            "committed_elastic_shell_layer_state_plus_"
                            "elastic_transverse_shear"
                        )
                    )
                ),
                "equivalent_stress": (
                    "committed_material_axis_hill48_layer_state"
                    if shell_uses_hill
                    else "conventional_von_mises"
                ),
                "mixed_reconstruction_von_mises": (
                    "committed_shell_layer_state_plus_elastic_transverse_shear"
                ),
                "stress_frame": (
                    "current_corotated_center_frame"
                    if str(kinematics) == "corotational"
                    else "reference_center_frame"
                ),
            },
        )
    beam, beam_reason = _state_beam_stresses(
        model, element, state, contextual_elastic
    )
    if beam is not None:
        beam_material = model.get_material(element.material_name)
        beam_material_history = _state_uses_plastic_constitutive_history(
            beam_material,
            state,
        )
        from .materials import is_orthotropic_material

        beam_is_orthotropic = is_orthotropic_material(beam_material)
        beam_uses_hill = (
            beam.get("equivalent_stress_measure") == "hill48"
        )
        return (
            beam,
            "committed_beam_fiber_state",
            "",
            {
                "axial_fibers_and_section_resultants": "committed_beam_fiber_state",
                "shear_and_torsion": "elastic_reconstruction_from_same_solution",
                "von_mises": (
                    "committed_beam_fiber_state_plus_elastic_shear_and_torsion"
                    if beam_is_orthotropic
                    else (
                        "committed_beam_fiber_state"
                        if beam_material_history
                        else (
                            "committed_elastic_beam_fiber_state_plus_"
                            "elastic_shear_and_torsion"
                        )
                    )
                ),
                "equivalent_stress": (
                    "committed_longitudinal_fiber_hill48_state"
                    if beam_uses_hill
                    else "conventional_von_mises"
                ),
                "mixed_reconstruction_von_mises": (
                    "committed_beam_fiber_state_plus_elastic_shear_and_torsion"
                ),
                "stress_frame": (
                    "current_corotated_frame"
                    if str(kinematics) == "corotational"
                    else "reference_frame"
                ),
            },
        )
    if reason == "element is not a shell":
        reason = beam_reason
    return None, "", reason, {}


def _coerce_element_states(
    *,
    nonlinear_result: Optional[Any],
    element_states: Optional[Mapping[int, Any]],
) -> Tuple[Mapping[int, Any], str, Tuple[str, ...]]:
    if nonlinear_result is not None and element_states is not None:
        raise ValueError("provide nonlinear_result or element_states, not both")
    if element_states is not None:
        return element_states, "explicit_element_states", ()
    if nonlinear_result is None:
        return {}, "none", ()
    states = getattr(nonlinear_result, "element_states", None)
    if isinstance(states, Mapping):
        return states, f"{type(nonlinear_result).__name__}.element_states", ()
    return {}, f"{type(nonlinear_result).__name__}", (
        "The supplied nonlinear result has no committed element_states; "
        "stress recovery is elastic-only.",
    )


def _recovery_analysis_context(
    nonlinear_result: Optional[Any],
    requested_kinematics: Optional[str],
) -> Tuple[str, Dict[str, Any]]:
    result_info = getattr(nonlinear_result, "info", None)
    info = result_info if isinstance(result_info, Mapping) else {}
    result_kinematics = info.get("kinematics")
    requested = (
        None
        if requested_kinematics is None
        else str(requested_kinematics).strip().lower()
    )
    inferred = (
        None
        if result_kinematics is None
        else str(result_kinematics).strip().lower()
    )
    if requested is not None and inferred is not None and requested != inferred:
        raise ValueError(
            "requested recovery kinematics does not match nonlinear_result.info"
        )
    kinematics = requested or inferred or "von_karman"
    if kinematics not in {"von_karman", "corotational"}:
        raise ValueError("recovery kinematics must be 'von_karman' or 'corotational'")
    context: Dict[str, Any] = {"kinematics": kinematics}
    if nonlinear_result is not None:
        context["result_type"] = type(nonlinear_result).__name__
        for key in ("status", "load_factor", "peak_load_factor"):
            value = getattr(nonlinear_result, key, None)
            if value is not None:
                context[key] = (
                    float(value)
                    if isinstance(value, (int, float, np.number))
                    else str(value)
                )
    return kinematics, context


def _recovery_displacements(
    model: "FEModel",
    displacements: Optional[np.ndarray],
    nonlinear_result: Optional[Any],
) -> np.ndarray:
    result_values = (
        None
        if nonlinear_result is None
        else getattr(nonlinear_result, "displacements", None)
    )
    supplied = (
        None
        if displacements is None
        else np.asarray(displacements, dtype=float).reshape(-1)
    )
    from_result = (
        None
        if result_values is None
        else np.asarray(result_values, dtype=float).reshape(-1)
    )
    if supplied is None and from_result is None:
        raise ValueError(
            "displacements are required unless nonlinear_result exposes them"
        )
    if supplied is not None and from_result is not None:
        if supplied.shape != from_result.shape or not np.allclose(
            supplied,
            from_result,
            rtol=1.0e-12,
            atol=1.0e-14,
        ):
            raise ValueError(
                "supplied displacements do not match nonlinear_result.displacements"
            )
    values = from_result if supplied is None else supplied
    expected = int(model.mesh.dof_manager.total_dofs)
    if values is None or values.shape != (expected,):
        actual = None if values is None else values.shape
        raise ValueError(
            f"displacements must have shape ({expected},), got {actual}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("displacements contain non-finite values")
    return values.copy()


def recover_stress_result(
    model: "FEModel",
    displacements: Optional[np.ndarray] = None,
    recovery_config: Optional[RecoveryConfig] = None,
    *,
    nonlinear_result: Optional[Any] = None,
    element_states: Optional[Mapping[int, Any]] = None,
    kinematics: Optional[str] = None,
    return_global: bool = False,
    resource_config: Optional[ResourceConfig] = None,
    patch_config: Optional[PatchRecoveryConfig] = None,
    copy_committed_states: bool = True,
) -> StressRecoveryResult:
    """Recover stresses with material-history provenance.

    ``nonlinear_result`` may be either a
    :class:`~anyfesolver.nonlinear_static.NonlinearStaticResult` or an
    :class:`~anyfesolver.arc_length.ArcLengthResult`; any object exposing
    ``element_states`` follows the same contract.  Directly supplied
    ``element_states`` are also supported.  Valid committed shell-layer,
    beam-fiber, and generalized shell/beam section states replace the
    corresponding elastic displacement reconstruction.  Generalized states
    retain their exact nonlinear section strains/resultants and never invent
    ply or fiber stresses.  Missing or invalid states are retained as
    explicitly labelled elastic fallbacks.

    Passing ``patch_config`` additionally performs guarded shell patch
    recovery from these unified integration-point stresses.  Therefore a
    yielded element is never silently replaced by elastic stresses in the
    patch samples.  Patch recovery is rejected for pre-integrated generalized
    shell sections because they do not define physical top/bottom stresses.

    When a nonlinear result is supplied, its displacement vector is used by
    default.  Passing a separate vector is allowed only when it matches the
    result, preventing committed material state from being combined with a
    different kinematic state.

    The returned result owns a deep-copied state snapshot by default.  Internal
    display-only consumers may set ``copy_committed_states=False`` to avoid
    duplicating a large state database; the returned snapshot then shares the
    caller's state objects and must be treated as read-only.
    """

    recovery = default_recovery_config(recovery_config)
    selected_ids = _ordered_element_ids(
        model,
        recovery.selected_element_ids(model),
    )
    return _recover_stress_result_after_selection(
        model,
        displacements,
        recovery=recovery,
        selected_ids=selected_ids,
        nonlinear_result=nonlinear_result,
        element_states=element_states,
        kinematics=kinematics,
        return_global=return_global,
        resource_config=resource_config,
        patch_config=patch_config,
        copy_committed_states=copy_committed_states,
    )


def _is_qualified_s3(element: Any) -> bool:
    """Return whether ``element`` is the qualified companion, never legacy TRI3."""

    from .e4_pl_s3_element import QualifiedE4PLS3ShellElement
    from .e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement

    return isinstance(
        element,
        (QualifiedE4PLS3ShellElement, NativeParityE4PLS3V2DShellElement),
    )


def _is_v2d_s3(element: Any) -> bool:
    from .e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement

    return (
        type(element) is NativeParityE4PLS3V2DShellElement
        and element.formulation_id == "CANDIDATE_E4_PL_S3_V2D_NATIVE_PARITY_V1"
    )


def _qualified_s3_current_physical_frame(
    reference_coordinates: np.ndarray,
    reference_frame: np.ndarray,
    current_coordinates: np.ndarray,
    current_directors: np.ndarray,
    director_polarity: int = 1,
) -> np.ndarray:
    """Reconstruct the objective S3 surface frame from committed geometry.

    This is the value-only counterpart of the rectangular right-polar frame
    differentiated by the native TL element.  It uses the reference numbered
    coordinates to define material tangent directions and the committed
    physical coordinates to transport them.  The stored director triads are
    an independent orientation witness, so an inverted or orthogonal surface
    fails closed instead of silently changing the top/bottom convention.
    """

    reference = np.asarray(reference_coordinates, dtype=float).reshape(3, 3)
    frame = np.asarray(reference_frame, dtype=float).reshape(3, 3)
    current = np.asarray(current_coordinates, dtype=float).reshape(3, 3)
    directors = np.asarray(current_directors, dtype=float).reshape(3, 3)
    if isinstance(director_polarity, (bool, np.bool_)) or not isinstance(
        director_polarity, (int, np.integer)
    ) or int(director_polarity) not in (-1, 1):
        raise ValueError("qualified S3 recovery director polarity must be -1 or +1")
    polarity = float(int(director_polarity))
    if not all(
        np.all(np.isfinite(values))
        for values in (reference, frame, current, directors)
    ):
        raise ValueError("qualified S3 recovery frame inputs must be finite")
    if not np.allclose(
        frame.T @ frame,
        np.eye(3, dtype=float),
        rtol=0.0,
        atol=1.0e-12,
    ) or float(np.linalg.det(frame)) <= 0.0:
        raise ValueError("qualified S3 recovery reference frame must be proper")

    local = (reference - reference[0]) @ frame[:, :2]
    reference_jacobian = np.array(
        (
            (local[1, 0] - local[0, 0], local[1, 1] - local[0, 1]),
            (local[2, 0] - local[0, 0], local[2, 1] - local[0, 1]),
        ),
        dtype=float,
    )
    determinant = float(np.linalg.det(reference_jacobian))
    if not math.isfinite(determinant) or abs(determinant) <= np.finfo(float).tiny:
        raise ValueError("qualified S3 recovery reference map is singular")
    inverse = np.linalg.inv(reference_jacobian)
    derivative_r = np.asarray((-1.0, 1.0, 0.0), dtype=float)
    derivative_s = np.asarray((-1.0, 0.0, 1.0), dtype=float)
    derivative_x = inverse[0, 0] * derivative_r + inverse[0, 1] * derivative_s
    derivative_y = inverse[1, 0] * derivative_r + inverse[1, 1] * derivative_s
    deformation = np.column_stack(
        (current.T @ derivative_x, current.T @ derivative_y)
    )
    metric_raw = deformation.T @ deformation
    metric = 0.5 * (metric_raw + metric_raw.T)
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    metric_determinant = float(np.linalg.det(metric))
    determinant_scale = max(abs(float(metric[0, 0] * metric[1, 1])), 1.0)
    if (
        not math.isfinite(metric_determinant)
        or metric_determinant
        <= 256.0 * np.finfo(float).eps * determinant_scale
        or float(eigenvalues[0]) <= 0.0
    ):
        raise ValueError("qualified S3 recovery current surface is singular")
    inverse_root = (
        eigenvectors
        @ np.diag(1.0 / np.sqrt(eigenvalues))
        @ eigenvectors.T
    )
    tangents = deformation @ inverse_root
    normal = np.cross(tangents[:, 0], tangents[:, 1])
    normal_norm = float(np.linalg.norm(normal))
    if not math.isfinite(normal_norm) or normal_norm <= np.finfo(float).tiny:
        raise ValueError("qualified S3 recovery current normal is singular")
    normal /= normal_norm
    physical_frame = np.column_stack((tangents[:, 0], tangents[:, 1], normal))
    if not np.allclose(
        physical_frame.T @ physical_frame,
        np.eye(3, dtype=float),
        rtol=0.0,
        atol=2.0e-11,
    ) or float(np.linalg.det(physical_frame)) <= 0.0:
        raise ValueError("qualified S3 recovery current frame is not proper")
    transported = np.sum(directors, axis=0)
    transported_norm = float(np.linalg.norm(transported))
    if (
        not math.isfinite(transported_norm)
        or transported_norm <= np.finfo(float).tiny
        or float(normal @ (polarity * transported / transported_norm)) <= 1.0e-8
    ):
        raise ValueError(
            "qualified S3 recovery current surface contradicts stored directors"
        )
    return physical_frame


def _qualified_s3_tensor_rows(values: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Rotate numbered-frame engineering tensor rows into global tensors."""

    rows = np.asarray(values, dtype=float).reshape(-1, 3)
    rotation = np.asarray(frame, dtype=float).reshape(3, 3)
    tensors = np.zeros((rows.shape[0], 3, 3), dtype=float)
    tensors[:, 0, 0] = rows[:, 0]
    tensors[:, 1, 1] = rows[:, 1]
    tensors[:, 0, 1] = rows[:, 2]
    tensors[:, 1, 0] = rows[:, 2]
    return np.asarray(
        [rotation @ tensor @ rotation.T for tensor in tensors],
        dtype=float,
    )


def _recover_v2d_committed_state(
    model: "FEModel",
    element_id: int,
    element: Any,
    state: Any,
    *,
    displacements: np.ndarray,
    return_global: bool,
) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, str]]:
    """Recover V2D Hammer-3 state without constitutive reevaluation."""

    if not _is_v2d_s3(element):
        return None, "element is not an S3 V2D candidate", {}
    if not isinstance(state, Mapping):
        return None, "S3 V2D committed state is not a mapping", {}
    material = model.get_material(element.material_name)
    try:
        local_u = np.asarray(
            element._get_element_displacements(model.mesh, displacements),
            dtype=np.float64,
        ).reshape(18)
        num_layers = int(state.get("num_layers", 0))
        committed = element.validate_model_bound_nonlinear_state(
            model.mesh,
            material,
            state,
            num_layers,
            expected_committed_total_u=local_u,
        )
    except (TypeError, ValueError, IndexError) as exc:
        return None, f"S3 V2D committed state validation failed: {exc}", {}

    try:
        from .corotational import (
            _corotational_cache,
            _corotational_deformation_state,
        )

        reference_coordinates = np.asarray(
            element.get_node_coordinates(model.mesh), dtype=np.float64
        ).reshape(3, 3)
        reference_record = _corotational_cache(model).get(int(element_id))
        if reference_record is None:
            raise ValueError("missing element-independent corotational reference")
        _deformational, rigid_rotation, current_coordinates = (
            _corotational_deformation_state(
                reference_record,
                element,
                local_u,
            )
        )
        reference_frame = np.asarray(
            element._geometry(model.mesh)["frame"], dtype=np.float64
        ).reshape(3, 3)
        current_frame = np.asarray(
            rigid_rotation @ reference_frame, dtype=np.float64
        )
        if committed["solver_kinematics"] == "VON_KARMAN":
            # Candidate VK resultants are work-conjugate to strains in the
            # fixed reference chart, not an EICR corotated material chart.
            current_frame = reference_frame.copy()
        physical_director = (
            float(element.director_polarity) * current_frame[:, 2]
        )
        reference_offset = float(element.reference_surface_offset)
        material_current = (
            np.asarray(current_coordinates, dtype=np.float64)
            - reference_offset * physical_director[None, :]
        )
    except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
        return None, f"S3 V2D current recovery frame is invalid: {exc}", {}

    strain = np.asarray(
        committed["station_generalized_strain"], dtype=np.float64
    ).reshape(3, 8)
    resultant = np.asarray(
        committed["station_generalized_resultant"], dtype=np.float64
    ).reshape(3, 8)
    recovered: Dict[str, Any] = {
        "formulation_id": str(element.formulation_id),
        "implementation_id": str(element.implementation_id),
        "recovery_scope": (
            "PHYSICAL_GLOBAL_RESULTANTS_ONLY"
            if element.shell_section is not None
            else "PHYSICAL_COMMITTED_GLOBAL_SURFACE_TENSORS"
        ),
        "physical_stress_available": element.shell_section is None,
        "physical_layer_recovery_available": element.shell_section is None,
        "membrane_strain": strain[:, :3].copy(),
        "curvature": strain[:, 3:6].copy(),
        "transverse_shear_strain": strain[:, 6:].copy(),
        "membrane_resultants": resultant[:, :3].copy(),
        "bending_resultants": resultant[:, 3:6].copy(),
        "transverse_shear_resultants": resultant[:, 6:].copy(),
        "numerical_fields_excluded": True,
        "recovery_history_source": "committed_s3_v2d_hammer3_state",
        "reference_surface_offset": reference_offset,
        "section_origin_offset_from_reference": -reference_offset,
        "physical_bottom_offset_from_reference": (
            -0.5 * float(element.thickness) - reference_offset
        ),
        "physical_top_offset_from_reference": (
            0.5 * float(element.thickness) - reference_offset
        ),
        "_reference_surface_recovery_coordinates": np.asarray(
            current_coordinates, dtype=np.float64
        ).copy(),
        "_recovery_coordinates": material_current.copy(),
        "_recovery_stress_frame": current_frame.copy(),
    }
    if return_global:
        recovered.update(
            {
                "global_membrane_resultant_tensors": _qualified_s3_tensor_rows(
                    resultant[:, :3], current_frame
                ),
                "global_bending_resultant_tensors": _qualified_s3_tensor_rows(
                    resultant[:, 3:6], current_frame
                ),
                "global_transverse_shear_resultants": (
                    resultant[:, 6, None] * current_frame[:, 0][None, :]
                    + resultant[:, 7, None] * current_frame[:, 1][None, :]
                ),
            }
        )
    component_sources = {
        "generalized_membrane_bending": "committed_s3_v2d_hammer3_state",
        "generalized_transverse_shear": "committed_s3_v2d_hammer3_state",
        "numerical_pl": "excluded_from_physical_recovery",
        "stress_frame": "element_independent_corotational_physical_frame",
    }
    if committed["solver_kinematics"] == "VON_KARMAN":
        component_sources["stress_frame"] = "experimental_von_karman_reference_frame"
    if element.shell_section is not None:
        recovered["generalized_stress_scope"] = "section_resultants_only"
        recovered["initial_field_provenance"] = copy.deepcopy(
            committed["initial_field_provenance"]
        )
        recovered["initial_generalized_prestrain"] = np.asarray(
            committed["initial_generalized_prestrain"], dtype=np.float64
        ).copy()
        recovered["initial_generalized_resultant"] = np.asarray(
            committed["initial_generalized_resultant"], dtype=np.float64
        ).copy()
        component_sources["physical_stress"] = (
            "unavailable_from_preintegrated_section"
        )
        return recovered, "", component_sources

    thickness = abs(float(element.thickness))
    if not math.isfinite(thickness) or thickness <= np.finfo(float).tiny:
        return None, "S3 V2D recovery thickness is invalid", {}
    try:
        layers = np.asarray(committed["layer_stress"], dtype=np.float64).reshape(
            3, num_layers, 3
        )
    except (TypeError, ValueError) as exc:
        return None, f"S3 V2D layer stress is malformed: {exc}", {}
    from .e4_pl_s3_state import qualified_s3_lobatto_layers

    layer_coordinates, _weights = qualified_s3_lobatto_layers(
        num_layers, thickness
    )
    recovered["physical_layer_coordinates_from_section_origin"] = (
        layer_coordinates.copy()
    )
    recovered["physical_layer_offsets_from_reference"] = (
        layer_coordinates - reference_offset
    )
    shear = resultant[:, 6:] / thickness
    in_plane_vm = np.sqrt(
        np.maximum(
            layers[:, :, 0] ** 2
            - layers[:, :, 0] * layers[:, :, 1]
            + layers[:, :, 1] ** 2
            + 3.0 * layers[:, :, 2] ** 2,
            0.0,
        )
    )
    mixed_vm = np.sqrt(
        np.maximum(
            in_plane_vm**2
            + 3.0 * (shear[:, 0, None] ** 2 + shear[:, 1, None] ** 2),
            0.0,
        )
    )
    recovered["in_plane_von_mises"] = np.max(in_plane_vm, axis=1)
    recovered["von_mises"] = np.max(mixed_vm, axis=1)
    recovered["equivalent_stress"] = recovered["von_mises"].copy()
    recovered["equivalent_stress_measure"] = "von_mises"
    for surface, layer_index in (("bot", 0), ("top", -1)):
        local_tensors = np.zeros((3, 3, 3), dtype=np.float64)
        local_tensors[:, 0, 0] = layers[:, layer_index, 0]
        local_tensors[:, 1, 1] = layers[:, layer_index, 1]
        local_tensors[:, 0, 1] = layers[:, layer_index, 2]
        local_tensors[:, 1, 0] = layers[:, layer_index, 2]
        local_tensors[:, 0, 2] = shear[:, 0]
        local_tensors[:, 2, 0] = shear[:, 0]
        local_tensors[:, 1, 2] = shear[:, 1]
        local_tensors[:, 2, 1] = shear[:, 1]
        global_tensors = np.asarray(
            [current_frame @ tensor @ current_frame.T for tensor in local_tensors],
            dtype=np.float64,
        )
        for first, second, label in (
            (0, 0, "xx"),
            (1, 1, "yy"),
            (2, 2, "zz"),
            (0, 1, "xy"),
            (1, 2, "yz"),
            (0, 2, "xz"),
        ):
            recovered[f"local_{label}_{surface}"] = local_tensors[
                :, first, second
            ].copy()
            if return_global:
                recovered[f"global_{label}_{surface}"] = global_tensors[
                    :, first, second
                ].copy()
    component_sources["physical_stress"] = (
        "committed_s3_v2d_layer_state_and_hammer3_resultants"
    )
    return recovered, "", component_sources


def _recover_qualified_s3_committed_state(
    model: "FEModel",
    element_id: int,
    element: Any,
    state: Any,
    *,
    displacements: np.ndarray,
    return_global: bool,
) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, str]]:
    """Recover one model-bound native S3 state without mechanics re-evaluation."""

    if _is_v2d_s3(element):
        return _recover_v2d_committed_state(
            model,
            element_id,
            element,
            state,
            displacements=displacements,
            return_global=return_global,
        )
    if not _is_qualified_s3(element):
        return None, "element is not a qualified S3", {}
    if not isinstance(state, Mapping):
        return None, "qualified S3 committed state is not a mapping", {}
    material = model.get_material(element.material_name)
    try:
        local_u = np.asarray(
            element._get_element_displacements(model.mesh, displacements),
            dtype=np.float64,
        ).reshape(18)
    except (TypeError, ValueError, IndexError) as exc:
        return None, f"qualified S3 committed displacement is invalid: {exc}", {}
    if element.shell_section is None:
        try:
            layer_stress_raw = np.asarray(
                state.get("layer_stress", ()), dtype=float
            )
        except (TypeError, ValueError) as exc:
            return (
                None,
                f"qualified S3 layer stress is malformed: {exc}",
                {},
            )
        if layer_stress_raw.size == 0 or layer_stress_raw.size % (7 * 3):
            return None, "qualified S3 state lacks seven-station layer stress", {}
        num_layers = int(layer_stress_raw.size // (7 * 3))
    else:
        # Stateless generalized sections persist station resultants but, by
        # contract, own no physical through-thickness layers.  The element
        # validator requires a positive solver count only to keep its public
        # signature uniform; it does not use that count for the V4 schema.
        num_layers = 1
    try:
        committed = element.validate_model_bound_nonlinear_state(
            model.mesh,
            material,
            state,
            num_layers,
            expected_committed_total_u=local_u,
        )
    except (TypeError, ValueError) as exc:
        return None, f"qualified S3 committed state validation failed: {exc}", {}

    reference = np.asarray(
        element.get_node_coordinates(model.mesh), dtype=np.float64
    ).reshape(3, 3)
    committed_u = np.asarray(
        committed["committed_total_u"], dtype=np.float64
    ).reshape(3, 6)
    current = reference + committed_u[:, :3]
    try:
        from .e4_pl_s3_element import triangle_frame

        reference_frame, _local, _quality = triangle_frame(
            reference,
            element.reference_normal,
        )
        triads = np.asarray(
            committed["committed_director_triads"], dtype=np.float64
        ).reshape(4, 3, 3)
        physical_directors = triads[:3, :, 2]
        reference_offset = float(
            getattr(element, "reference_surface_offset", 0.0)
        )
        reference_directors = np.repeat(
            (
                float(element.director_polarity)
                * np.asarray(reference_frame[:, 2], dtype=np.float64)
            )[None, :],
            3,
            axis=0,
        )
        material_reference = reference - reference_offset * reference_directors
        material_current = current - reference_offset * physical_directors
        current_frame = _qualified_s3_current_physical_frame(
            material_reference,
            reference_frame,
            material_current,
            physical_directors,
            int(element.director_polarity),
        )
    except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
        return None, f"qualified S3 current recovery frame is invalid: {exc}", {}

    strain = np.asarray(
        committed["station_generalized_strain"], dtype=np.float64
    ).reshape(7, 8)
    resultant = np.asarray(
        committed["station_generalized_resultant"], dtype=np.float64
    ).reshape(7, 8)
    from .shell_sections import (
        SHELL_MEMBRANE_VOIGT_ORDER,
        SHELL_TRANSVERSE_SHEAR_ORDER,
    )

    recovered: Dict[str, Any] = {
        "recovery_scope": (
            "section_resultants_only"
            if element.shell_section is not None
            else "qualified_s3_committed_native_physical"
        ),
        "physical_stress_available": element.shell_section is None,
        "physical_layer_recovery_available": element.shell_section is None,
        "membrane_resultant_order": SHELL_MEMBRANE_VOIGT_ORDER,
        "transverse_shear_resultant_order": SHELL_TRANSVERSE_SHEAR_ORDER,
        "membrane_strain": strain[:, :3].copy(),
        "curvature": strain[:, 3:6].copy(),
        "transverse_shear_strain": strain[:, 6:].copy(),
        "membrane_resultants": resultant[:, :3].copy(),
        "bending_resultants": resultant[:, 3:6].copy(),
        "transverse_shear_resultants": resultant[:, 6:].copy(),
        "numerical_fields_excluded": True,
        "recovery_history_source": "committed_s3_station7_state",
        "reference_surface_offset": reference_offset,
        "section_origin_offset_from_reference": -reference_offset,
        "physical_bottom_offset_from_reference": (
            -0.5 * float(element.thickness) - reference_offset
        ),
        "physical_top_offset_from_reference": (
            0.5 * float(element.thickness) - reference_offset
        ),
        "_reference_surface_recovery_coordinates": current.copy(),
        "_recovery_coordinates": material_current.copy(),
        "_recovery_stress_frame": current_frame.copy(),
    }
    if return_global:
        recovered.update(
            {
                "global_membrane_resultant_tensors": (
                    _qualified_s3_tensor_rows(resultant[:, :3], current_frame)
                ),
                "global_bending_resultant_tensors": (
                    float(element.director_polarity)
                    * _qualified_s3_tensor_rows(resultant[:, 3:6], current_frame)
                ),
                "global_transverse_shear_resultants": (
                    float(element.director_polarity)
                    * (
                        resultant[:, 6, None] * current_frame[:, 0][None, :]
                        + resultant[:, 7, None] * current_frame[:, 1][None, :]
                    )
                ),
            }
        )

    component_sources = {
        "generalized_membrane_bending": "committed_s3_station7_state",
        "generalized_transverse_shear": "committed_s3_station7_state",
        "numerical_pl": "excluded_from_physical_recovery",
        "stress_frame": "committed_s3_objective_current_physical_frame",
    }
    if element.shell_section is not None:
        recovered["generalized_stress_scope"] = "section_resultants_only"
        recovered.update(
            {
                "generalized_initial_field_policy_id": committed[
                    "generalized_initial_field_policy_id"
                ],
                "initial_fields_fingerprint": committed[
                    "initial_fields_fingerprint"
                ],
                "initial_field_provenance": copy.deepcopy(
                    committed["initial_field_provenance"]
                ),
                "initial_generalized_prestrain": np.asarray(
                    committed["initial_generalized_prestrain"],
                    dtype=np.float64,
                ).copy(),
                "initial_generalized_resultant": np.asarray(
                    committed["initial_generalized_resultant"],
                    dtype=np.float64,
                ).copy(),
            }
        )
        component_sources["physical_stress"] = (
            "unavailable_from_preintegrated_section"
        )
        component_sources["initial_fields"] = (
            "committed_s3_generalized_initial_fields"
        )
        return recovered, "", component_sources

    thickness = abs(float(element.thickness))
    if not math.isfinite(thickness) or thickness <= np.finfo(float).tiny:
        return None, "qualified S3 recovery thickness is invalid", {}
    layers = np.asarray(committed["layer_stress"], dtype=float).reshape(
        7, num_layers, 3
    )
    from .e4_pl_s3_state import qualified_s3_lobatto_layers

    layer_coordinates, _layer_weights = qualified_s3_lobatto_layers(
        num_layers,
        thickness,
    )
    recovered["physical_layer_coordinates_from_section_origin"] = (
        layer_coordinates.copy()
    )
    recovered["physical_layer_offsets_from_reference"] = (
        layer_coordinates - reference_offset
    )
    shear = resultant[:, 6:] / thickness
    membrane = resultant[:, :3] / thickness
    bending = 6.0 * resultant[:, 3:6] / (thickness * thickness)
    recovered.update(
        {
            "membrane_xx": membrane[:, 0].copy(),
            "membrane_yy": membrane[:, 1].copy(),
            "membrane_xy": membrane[:, 2].copy(),
            "bending_xx": bending[:, 0].copy(),
            "bending_yy": bending[:, 1].copy(),
            "bending_xy": bending[:, 2].copy(),
            "shear_xz": shear[:, 0].copy(),
            "shear_yz": shear[:, 1].copy(),
        }
    )
    in_plane_vm = np.sqrt(
        np.maximum(
            layers[:, :, 0] ** 2
            - layers[:, :, 0] * layers[:, :, 1]
            + layers[:, :, 1] ** 2
            + 3.0 * layers[:, :, 2] ** 2,
            0.0,
        )
    )
    mixed_vm = np.sqrt(
        np.maximum(
            in_plane_vm**2
            + 3.0
            * (shear[:, 0, None] ** 2 + shear[:, 1, None] ** 2),
            0.0,
        )
    )
    from .materials import is_orthotropic_material

    material_history_active = _state_uses_plastic_constitutive_history(
        material,
        committed,
    )
    primary_vm = (
        mixed_vm
        if is_orthotropic_material(material)
        else (in_plane_vm if material_history_active else mixed_vm)
    )
    recovered["in_plane_von_mises"] = np.max(in_plane_vm, axis=1)
    recovered["von_mises"] = np.max(primary_vm, axis=1)
    recovered["mixed_reconstruction_von_mises"] = np.max(mixed_vm, axis=1)
    hill_yield = getattr(material, "hill_yield", None)
    if hill_yield is not None:
        from .plasticity import hill48_plane_stress_equivalent_stress

        material_layers = np.asarray(
            committed["layer_stress_material"], dtype=float
        ).reshape(7, num_layers, 3)
        hill_equivalent = hill48_plane_stress_equivalent_stress(
            material_layers.reshape(-1, 3),
            hill_yield,
        ).reshape(7, num_layers)
        alpha = np.asarray(committed["alpha"], dtype=float).reshape(
            7, num_layers
        )
        current_strength = _hill_current_x_strength(
            material,
            alpha.reshape(-1),
        ).reshape(7, num_layers)
        recovered["equivalent_stress"] = np.max(hill_equivalent, axis=1)
        recovered["hill_utilization"] = np.max(
            hill_equivalent
            / np.maximum(current_strength, np.finfo(float).tiny),
            axis=1,
        )
        recovered["equivalent_stress_measure"] = "hill48"
        recovered["equivalent_stress_scope"] = (
            "committed_s3_material_axis_plane_stress_layers"
        )
    else:
        recovered["equivalent_stress"] = recovered["von_mises"].copy()
        recovered["equivalent_stress_measure"] = "von_mises"

    for surface, layer_index in (("bot", 0), ("top", -1)):
        local_tensors = np.zeros((7, 3, 3), dtype=float)
        local_tensors[:, 0, 0] = layers[:, layer_index, 0]
        local_tensors[:, 1, 1] = layers[:, layer_index, 1]
        local_tensors[:, 0, 1] = layers[:, layer_index, 2]
        local_tensors[:, 1, 0] = layers[:, layer_index, 2]
        owner_shear = float(element.director_polarity) * shear
        local_tensors[:, 0, 2] = owner_shear[:, 0]
        local_tensors[:, 2, 0] = owner_shear[:, 0]
        local_tensors[:, 1, 2] = owner_shear[:, 1]
        local_tensors[:, 2, 1] = owner_shear[:, 1]
        global_tensors = np.asarray(
            [current_frame @ tensor @ current_frame.T for tensor in local_tensors],
            dtype=float,
        )
        for first, second, label in (
            (0, 0, "xx"),
            (1, 1, "yy"),
            (2, 2, "zz"),
            (0, 1, "xy"),
            (1, 2, "yz"),
            (0, 2, "xz"),
        ):
            recovered[f"local_{label}_{surface}"] = local_tensors[
                :, first, second
            ].copy()
            if return_global:
                recovered[f"global_{label}_{surface}"] = global_tensors[
                    :, first, second
                ].copy()
    component_sources.update(
        {
            "membrane_bending_and_surface_in_plane": (
                "committed_s3_layer_state_and_station7_resultants"
            ),
            "transverse_shear": "committed_s3_station7_resultants",
            "von_mises": (
                "committed_s3_layer_state_plus_station7_transverse_shear"
                if is_orthotropic_material(material)
                or not material_history_active
                else "committed_s3_layer_state_in_plane"
            ),
            "mixed_reconstruction_von_mises": (
                "committed_s3_layer_state_plus_station7_transverse_shear"
            ),
            "physical_stress": "committed_s3_layer_state",
        }
    )
    return recovered, "", component_sources


def _recover_stress_result_after_selection(
    model: "FEModel",
    displacements: Optional[np.ndarray],
    *,
    recovery: RecoveryConfig,
    selected_ids: Tuple[int, ...],
    nonlinear_result: Optional[Any],
    element_states: Optional[Mapping[int, Any]],
    kinematics: Optional[str],
    return_global: bool,
    resource_config: Optional[ResourceConfig],
    patch_config: Optional[PatchRecoveryConfig],
    copy_committed_states: bool,
) -> StressRecoveryResult:
    """Complete unified recovery after deterministic element selection."""

    if recovery.include_stresses and (
        kinematics is not None or nonlinear_result is not None
    ):
        require_model_element_capabilities(
            model,
            "nonlinear_geometry",
            context="recover_stress_result",
            element_ids=selected_ids,
        )
    required_capabilities: set[str] = set()
    if return_global and recovery.include_stresses:
        required_capabilities.add("global_recovery")
    if patch_config is not None:
        required_capabilities.update(("global_recovery", "patch_recovery"))
    if required_capabilities:
        require_model_element_capabilities(
            model,
            required_capabilities,
            context="recover_stress_result",
            element_ids=selected_ids,
        )
    states, state_source, initial_warnings = _coerce_element_states(
        nonlinear_result=nonlinear_result,
        element_states=element_states,
    )
    resolved_kinematics, analysis_context = _recovery_analysis_context(
        nonlinear_result,
        kinematics,
    )
    result_info = getattr(nonlinear_result, "info", {})
    stable_shell_rotation = (result_info.get("corotational_shell_rotation_stable", False)
                             if isinstance(result_info, Mapping) else False)
    if type(stable_shell_rotation) is not bool:
        raise ValueError("invalid corotational shell rotation precision provenance")
    analysis_context["corotational_shell_rotation_stable"] = stable_shell_rotation
    selected_set = set(selected_ids)
    selected_state_ids = tuple(
        sorted(
            int(element_id)
            for element_id in states
            if int(element_id) in selected_set
        )
    )
    if selected_state_ids:
        require_model_element_capabilities(
            model,
            "committed_state_recovery",
            context="recover_stress_result",
            element_ids=selected_state_ids,
        )
    selected_states = {
        int(element_id): state
        for element_id, state in sorted(states.items(), key=lambda item: int(item[0]))
        if int(element_id) in selected_set
    }
    state_snapshot = (
        copy.deepcopy(selected_states) if copy_committed_states else dict(selected_states)
    )
    if not recovery.include_stresses:
        _disabled_stresses, report = recover_element_stresses_with_report(
            model,
            displacements,
            recovery,
            return_global=bool(return_global),
            resource_config=resource_config,
        )
        provenance = StressRecoveryProvenance(
            mode="disabled",
            state_source=state_source,
            analysis_context=analysis_context,
            return_global=bool(return_global),
            warnings=initial_warnings,
        )
        return StressRecoveryResult(
            {},
            provenance,
            state_snapshot,
            report,
            None,
        )
    displacement_values = _recovery_displacements(
        model,
        displacements,
        nonlinear_result,
    )

    # A valid generalized-section state already contains the exact nonlinear
    # constitutive quantities.  Recover it before the elastic pass so von
    # Karman terms and corotational basic forces cannot be overwritten by a
    # call to the linear ``compute_stresses`` implementation.
    committed_generalized: Dict[
        int,
        Tuple[Dict[str, Any], str, Dict[str, str]],
    ] = {}
    generalized_failures: Dict[int, str] = {}
    generalized_return_global = bool(return_global or patch_config is not None)
    for element_id, state in selected_states.items():
        element = model.mesh.elements.get(int(element_id))
        if element is None:
            continue
        if _is_qualified_s3(element):
            state_stress, reason, component_sources = (
                _recover_qualified_s3_committed_state(
                    model,
                    int(element_id),
                    element,
                    state,
                    displacements=displacement_values,
                    return_global=generalized_return_global,
                )
            )
            if state_stress is None:
                raise ValueError(
                    "qualified S3 committed recovery failed closed for "
                    f"element {int(element_id)}: {reason}"
                )
            committed_generalized[int(element_id)] = (
                state_stress,
                (
                    "committed_s3_v2d_hammer3_state"
                    if _is_v2d_s3(element)
                    else "committed_qualified_s3_native_state"
                ),
                component_sources,
            )
            continue
        if not _element_has_generalized_section(element):
            continue
        state_stress, source, reason, component_sources = (
            _recover_generalized_committed_state(
                model,
                int(element_id),
                state,
                displacements=displacement_values,
                kinematics=resolved_kinematics,
                return_global=generalized_return_global,
            )
        )
        if state_stress is not None:
            committed_generalized[int(element_id)] = (
                state_stress,
                source,
                component_sources,
            )
        else:
            generalized_failures[int(element_id)] = (
                reason or "unsupported committed generalized-section state"
            )

    unfiltered_recovery = replace(recovery, components=None)
    elastic_recovery_ids = tuple(
        element_id
        for element_id in selected_ids
        if element_id not in committed_generalized
    )
    elastic_recovery = replace(
        unfiltered_recovery,
        element_ids=elastic_recovery_ids,
    )
    elastic_stresses, report = recover_element_stresses_with_report(
        model,
        displacement_values,
        elastic_recovery,
        return_global=bool(return_global or patch_config is not None),
        resource_config=resource_config,
    )
    if resolved_kinematics == "corotational":
        for element_id in elastic_recovery_ids:
            element = model.mesh.elements.get(int(element_id))
            if element is None or not callable(
                getattr(element, "compute_stresses", None)
            ):
                continue
            contextual, _frame, current_coords = _element_recovery_context(
                model,
                int(element_id),
                element,
                displacement_values,
                kinematics=resolved_kinematics,
                return_global=bool(return_global or patch_config is not None),
                stable_shell_rotation=stable_shell_rotation,
            )
            contextual["_recovery_coordinates"] = np.asarray(
                current_coords, dtype=float
            ).copy()
            elastic_stresses[int(element_id)] = contextual
    recovered: Dict[int, Dict[str, Any]] = {}
    per_element_source: Dict[int, str] = {}
    per_element_component_sources: Dict[int, Dict[str, str]] = {}
    fallback_reasons: Dict[int, str] = {}
    warnings = list(initial_warnings)
    for element_id in selected_ids:
        if int(element_id) in committed_generalized:
            state_stress, source, component_sources = committed_generalized[
                int(element_id)
            ]
            recovered[int(element_id)] = state_stress
            per_element_source[int(element_id)] = source
            per_element_component_sources[int(element_id)] = (
                component_sources
            )
            continue
        elastic = elastic_stresses.get(int(element_id), {})
        if int(element_id) in selected_states:
            if int(element_id) in generalized_failures:
                state_stress = None
                source = ""
                component_sources = {}
                reason = generalized_failures[int(element_id)]
            else:
                state_stress, source, reason, component_sources = (
                    _recover_one_committed_state(
                        model,
                        int(element_id),
                        selected_states[int(element_id)],
                        elastic,
                        displacements=displacement_values,
                        kinematics=resolved_kinematics,
                        stable_shell_rotation=stable_shell_rotation,
                        return_global=bool(
                            return_global or patch_config is not None
                        ),
                    )
                )
            if state_stress is not None:
                recovered[int(element_id)] = state_stress
                per_element_source[int(element_id)] = source
                per_element_component_sources[int(element_id)] = (
                    component_sources
                )
                continue
            fallback_reasons[int(element_id)] = reason or "unsupported committed state"
        if elastic:
            recovered[int(element_id)] = dict(elastic)
            per_element_source[int(element_id)] = "elastic_displacement_reconstruction"
            per_element_component_sources[int(element_id)] = {
                "all_reported_components": "elastic_reconstruction_from_same_solution",
                "stress_frame": (
                    "current_corotated_frame"
                    if resolved_kinematics == "corotational"
                    else "reference_element_frame"
                ),
            }

    history_ids = tuple(
        element_id
        for element_id in selected_ids
        if per_element_source.get(element_id, "").startswith("committed_")
    )
    elastic_ids = tuple(
        element_id
        for element_id in selected_ids
        if per_element_source.get(element_id) == "elastic_displacement_reconstruction"
    )
    if history_ids and elastic_ids:
        mode = "mixed"
    elif history_ids:
        mode = "material_history"
    else:
        mode = "elastic_only"
    if fallback_reasons:
        warnings.append(
            "Some supplied committed states could not be recovered and are "
            "explicitly labelled elastic fallbacks."
        )

    generalized_shell_ids = tuple(
        int(element_id)
        for element_id in selected_ids
        if int(element_id) in recovered
        and getattr(
            model.mesh.elements.get(int(element_id)),
            "shell_section",
            None,
        )
        is not None
    )
    if patch_config is not None and generalized_shell_ids:
        raise ValueError(
            "shell patch recovery requires physical top/bottom surface "
            "stresses, which pre-integrated generalized shell sections do "
            "not provide; recover section resultants without patch_config"
        )

    nodal_stresses = None
    if patch_config is not None:
        nodal_stresses = recover_shell_patch_stresses(
            model,
            recovered,
            element_ids=selected_ids,
            config=patch_config,
            per_element_source=per_element_source,
        )

    components = recovery.selected_components()
    if components is not None:
        recovered = {
            int(element_id): _filter_components(values, components)
            for element_id, values in recovered.items()
        }
    if not return_global:
        # Global surface tensors are an internal prerequisite for patch
        # recovery, not an implicit expansion of the requested element API.
        recovered = {
            int(element_id): {
                key: value
                for key, value in values.items()
                if not str(key).startswith(
                    ("global_", "local_", "_recovery_")
                )
            }
            for element_id, values in recovered.items()
        }
    else:
        recovered = {
            int(element_id): {
                key: value
                for key, value in values.items()
                if not str(key).startswith("_recovery_")
            }
            for element_id, values in recovered.items()
        }

    provenance = StressRecoveryProvenance(
        mode=mode,
        state_source=state_source,
        per_element_source=per_element_source,
        per_element_component_sources=per_element_component_sources,
        history_aware_element_ids=history_ids,
        elastic_reconstruction_element_ids=elastic_ids,
        fallback_reasons=fallback_reasons,
        analysis_context=analysis_context,
        return_global=bool(return_global),
        warnings=tuple(warnings),
    )
    return StressRecoveryResult(
        element_stresses=recovered,
        provenance=provenance,
        committed_element_states=state_snapshot,
        execution_report=report,
        nodal_stresses=nodal_stresses,
    )


def _prepare_shell_patch_elements(
    model: "FEModel",
    element_stresses: Mapping[int, Mapping[str, Any]],
    element_ids: Optional[Sequence[int]],
    config: PatchRecoveryConfig,
) -> Tuple[
    Dict[int, Dict[str, Any]],
    Dict[int, str],
    Dict[int, Tuple[int, ...]],
]:
    from .elements import ShellElement
    from .results import _gauss_to_node_extrapolation
    from .e4_pl_element import QualifiedE4PLShellElement
    from .e4_pl_s3_element import QualifiedE4PLS3ShellElement
    from .e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement

    selected = (
        set(int(element_id) for element_id in element_ids)
        if element_ids is not None
        else set(int(element_id) for element_id in element_stresses)
    )
    prepared: Dict[int, Dict[str, Any]] = {}
    skipped: Dict[int, str] = {}
    all_shell_incidence: Dict[int, list[int]] = {}
    for element_id in sorted(selected):
        element = model.mesh.elements.get(int(element_id))
        if not isinstance(element, ShellElement):
            continue
        for node_id in element.node_ids:
            all_shell_incidence.setdefault(int(node_id), []).append(
                int(element_id)
            )
        qualified_s3 = isinstance(
            element,
            (QualifiedE4PLS3ShellElement, NativeParityE4PLS3V2DShellElement),
        )
        if int(element.num_nodes) not in {4, 8} and not (
            int(element.num_nodes) == 3 and qualified_s3
        ):
            skipped[int(element_id)] = "unsupported_shell_topology"
            continue
        if int(element.num_nodes) == 8 and bool(
            getattr(element, "reduced_integration", False)
        ):
            skipped[int(element_id)] = "unqualified_reduced_q8_topology"
            continue
        stresses = element_stresses.get(int(element_id))
        if not isinstance(stresses, Mapping):
            skipped[int(element_id)] = "missing_element_stresses"
            continue
        n_gp = int(len(element.gauss_points))
        arrays: Dict[str, np.ndarray] = {}
        valid = True
        for key in _SHELL_SURFACE_KEYS:
            values = np.asarray(stresses.get(key, ()), dtype=float).reshape(-1)
            if values.size != n_gp or not np.all(np.isfinite(values)):
                valid = False
                break
            arrays[key] = values
        if not valid:
            skipped[int(element_id)] = "missing_or_invalid_global_surface_stresses"
            continue
        operator = _gauss_to_node_extrapolation(element)
        if operator is None or operator.shape != (int(element.num_nodes), n_gp):
            skipped[int(element_id)] = "no_gauss_to_node_fallback_operator"
            continue
        coords = np.asarray(
            stresses.get(
                "_recovery_coordinates",
                element.get_node_coordinates(model.mesh),
            ),
            dtype=float,
        )
        if coords.shape != (int(element.num_nodes), 3) or not np.all(
            np.isfinite(coords)
        ):
            skipped[int(element_id)] = "invalid_recovery_coordinates"
            continue
        sample_positions = np.empty((n_gp, 3), dtype=float)
        try:
            for gp_index, (xi, eta) in enumerate(np.asarray(element.gauss_points, dtype=float)):
                shape, _dN_dxi, _dN_deta = element.compute_shape_functions(float(xi), float(eta))
                sample_positions[gp_index] = np.asarray(shape, dtype=float) @ coords
            stored_frame = np.asarray(
                stresses.get("_recovery_stress_frame", ()), dtype=float
            )
            if stored_frame.size:
                rotation = stored_frame.reshape(3, 3)
            elif qualified_s3:
                from .e4_pl_s3_element import triangle_frame

                rotation, _local, _quality = triangle_frame(
                    coords,
                    element.reference_normal,
                )
            else:
                rotation = np.asarray(element._center_frame(coords), dtype=float)
        except (ValueError, np.linalg.LinAlgError):
            skipped[int(element_id)] = "invalid_shell_geometry"
            continue
        if (
            not np.all(np.isfinite(rotation))
            or not np.allclose(
                rotation.T @ rotation,
                np.eye(3, dtype=float),
                rtol=0.0,
                atol=2.0e-11,
            )
            or float(np.linalg.det(rotation)) <= 0.0
        ):
            skipped[int(element_id)] = "invalid_recovery_stress_frame"
            continue
        centroid = np.mean(coords, axis=0)
        scale = max(
            float(np.max(np.linalg.norm(coords - centroid, axis=1))),
            np.finfo(float).tiny,
        )
        out_of_plane = np.abs((coords - centroid) @ rotation[:, 2])
        if float(np.max(out_of_plane)) > (
            float(config.planarity_relative_tolerance) * scale
        ):
            skipped[int(element_id)] = "warped_shell_outside_patch_scope"
            continue
        cosine_limit = math.cos(
            math.radians(float(config.normal_tolerance_degrees))
        )
        inconsistent_gp_frame = False
        if not qualified_s3:
            try:
                for xi, eta in np.asarray(element.gauss_points, dtype=float):
                    _shape, dN_dxi, dN_deta = element.compute_shape_functions(
                        float(xi), float(eta)
                    )
                    gp_rotation, _dx, _dy, _det = (
                        element._local_frame_and_derivatives(
                            coords, dN_dxi, dN_deta
                        )
                    )
                    if float(gp_rotation[:, 2] @ rotation[:, 2]) < cosine_limit:
                        inconsistent_gp_frame = True
                        break
            except (ValueError, np.linalg.LinAlgError):
                skipped[int(element_id)] = "invalid_shell_geometry"
                continue
        if inconsistent_gp_frame:
            skipped[int(element_id)] = "warped_shell_normal_variation"
            continue
        prepared[int(element_id)] = {
            "element": element,
            "patch_family": (
                "qualified_e4_pl_shell"
                if isinstance(
                    element,
                    (
                        QualifiedE4PLShellElement,
                        QualifiedE4PLS3ShellElement,
                        NativeParityE4PLS3V2DShellElement,
                    ),
                )
                else f"shell_{int(element.num_nodes)}"
            ),
            "coords": coords,
            "positions": sample_positions,
            "normal": rotation[:, 2].copy(),
            "tangent": rotation[:, 0].copy(),
            "values": np.column_stack([arrays[key] for key in _SHELL_SURFACE_KEYS]),
            "element_nodal": {
                key: operator @ arrays[key] for key in _SHELL_SURFACE_KEYS
            },
        }
    return (
        prepared,
        skipped,
        {
            int(node_id): tuple(sorted(element_ids_at_node))
            for node_id, element_ids_at_node in sorted(
                all_shell_incidence.items()
            )
        },
    )


def _patch_fallback_values(
    node_id: int,
    incident: Sequence[Tuple[int, Dict[str, Any]]],
) -> Optional[np.ndarray]:
    values = []
    for _element_id, data in incident:
        element = data["element"]
        try:
            local_index = tuple(int(value) for value in element.node_ids).index(int(node_id))
        except ValueError:
            continue
        values.append(
            np.array(
                [data["element_nodal"][key][local_index] for key in _SHELL_SURFACE_KEYS],
                dtype=float,
            )
        )
    if not values:
        return None
    return np.mean(np.vstack(values), axis=0)


def _patch_guard_reason(
    incident: Sequence[Tuple[int, Dict[str, Any]]],
    per_element_source: Mapping[int, str],
    config: PatchRecoveryConfig,
) -> Optional[str]:
    topologies = {int(data["element"].num_nodes) for _element_id, data in incident}
    patch_families = {
        str(data.get("patch_family", f"shell_{int(data['element'].num_nodes)}"))
        for _element_id, data in incident
    }
    if len(topologies) != 1 and patch_families != {"qualified_e4_pl_shell"}:
        return "mixed_shell_topology"
    materials = {str(data["element"].material_name) for _element_id, data in incident}
    if config.material_continuity_required and len(materials) != 1:
        return "material_discontinuity"
    thicknesses = np.array(
        [float(data["element"].thickness) for _element_id, data in incident],
        dtype=float,
    )
    scale = max(float(np.max(np.abs(thicknesses))), np.finfo(float).tiny)
    if float(np.ptp(thicknesses)) > float(config.thickness_relative_tolerance) * scale:
        return "thickness_discontinuity"
    sources = {
        str(per_element_source.get(int(element_id), "unspecified"))
        for element_id, _data in incident
    }
    if len(sources) != 1:
        return "mixed_recovery_source"
    return None


def _partition_patch_regions(
    incident: Sequence[Tuple[int, Dict[str, Any]]],
    per_element_source: Mapping[int, str],
    config: PatchRecoveryConfig,
) -> list[list[Tuple[int, Dict[str, Any]]]]:
    """Partition a nodal patch without ever crossing a discontinuity."""

    cosine_limit = math.cos(
        math.radians(float(config.normal_tolerance_degrees))
    )
    regions: list[list[Tuple[int, Dict[str, Any]]]] = []
    for item in sorted(incident, key=lambda pair: pair[0]):
        placed = False
        for region in regions:
            candidate = [*region, item]
            if _patch_guard_reason(
                candidate, per_element_source, config
            ) is not None:
                continue
            reference_normal = np.asarray(
                region[0][1]["normal"], dtype=float
            )
            candidate_normal = np.asarray(item[1]["normal"], dtype=float)
            if float(reference_normal @ candidate_normal) < cosine_limit:
                continue
            region.append(item)
            placed = True
            break
        if not placed:
            regions.append([item])
    return regions


def _patch_node_coordinates(
    node_id: int,
    incident: Sequence[Tuple[int, Dict[str, Any]]],
) -> Optional[np.ndarray]:
    for _element_id, data in incident:
        element = data["element"]
        try:
            local_index = tuple(int(value) for value in element.node_ids).index(
                int(node_id)
            )
        except ValueError:
            continue
        return np.asarray(data["coords"][local_index], dtype=float).copy()
    return None


def _fit_shell_patch(
    node_coords: np.ndarray,
    incident: Sequence[Tuple[int, Dict[str, Any]]],
    config: PatchRecoveryConfig,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    positions = np.vstack([data["positions"] for _element_id, data in incident])
    values = np.vstack([data["values"] for _element_id, data in incident])
    normals = np.vstack([data["normal"] for _element_id, data in incident])
    reference_normal = normals[0] / max(float(np.linalg.norm(normals[0])), np.finfo(float).tiny)
    cosine_limit = math.cos(math.radians(float(config.normal_tolerance_degrees)))
    alignment = normals @ reference_normal
    if np.any(alignment < cosine_limit):
        return None, {
            "status": "fallback",
            "reason": "inconsistent_or_curved_shell_normals",
            "sample_count": int(positions.shape[0]),
        }
    average_normal = np.sum(normals, axis=0)
    normal_norm = float(np.linalg.norm(average_normal))
    if normal_norm <= np.finfo(float).eps:
        return None, {
            "status": "fallback",
            "reason": "undefined_patch_normal",
            "sample_count": int(positions.shape[0]),
        }
    average_normal /= normal_norm
    tangent = np.asarray(incident[0][1]["tangent"], dtype=float)
    tangent -= float(np.dot(tangent, average_normal)) * average_normal
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= np.finfo(float).eps:
        return None, {
            "status": "fallback",
            "reason": "undefined_patch_tangent",
            "sample_count": int(positions.shape[0]),
        }
    tangent /= tangent_norm
    second_tangent = np.cross(average_normal, tangent)
    second_tangent /= max(float(np.linalg.norm(second_tangent)), np.finfo(float).tiny)

    offsets = positions - np.asarray(node_coords, dtype=float).reshape(1, 3)
    local_x = offsets @ tangent
    local_y = offsets @ second_tangent
    coordinate_scale = max(
        float(np.max(np.sqrt(local_x**2 + local_y**2))),
        np.finfo(float).tiny,
    )
    design = np.column_stack(
        [
            np.ones(positions.shape[0], dtype=float),
            local_x / coordinate_scale,
            local_y / coordinate_scale,
        ]
    )
    singular_values = np.linalg.svd(design, compute_uv=False)
    rank = int(np.linalg.matrix_rank(design))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size and singular_values[-1] > 0.0
        else float("inf")
    )
    diagnostics = {
        "sample_count": int(positions.shape[0]),
        "rank": rank,
        "condition_number": condition,
    }
    if positions.shape[0] < 3 or rank < 3:
        diagnostics.update(status="fallback", reason="rank_deficient_patch")
        return None, diagnostics
    if not np.isfinite(condition) or condition > float(config.condition_limit):
        diagnostics.update(status="fallback", reason="ill_conditioned_patch")
        return None, diagnostics
    coefficients, _residuals, fitted_rank, _singular = np.linalg.lstsq(
        design, values, rcond=None
    )
    if int(fitted_rank) < 3 or not np.all(np.isfinite(coefficients)):
        diagnostics.update(status="fallback", reason="patch_fit_failed")
        return None, diagnostics
    diagnostics.update(status="qualified", reason="")
    return coefficients[0].copy(), diagnostics


def _nodal_surface_mapping(values: np.ndarray) -> Dict[str, float]:
    mapped = {
        key: float(value) for key, value in zip(_SHELL_SURFACE_KEYS, values)
    }
    for surface in ("top", "bot"):
        sx = mapped[f"global_xx_{surface}"]
        sy = mapped[f"global_yy_{surface}"]
        sz = mapped[f"global_zz_{surface}"]
        txy = mapped[f"global_xy_{surface}"]
        tyz = mapped[f"global_yz_{surface}"]
        txz = mapped[f"global_xz_{surface}"]
        mapped[f"von_mises_{surface}"] = float(
            np.sqrt(
                max(
                    0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
                    + 3.0 * (txy**2 + tyz**2 + txz**2),
                    0.0,
                )
            )
        )
    mapped["von_mises"] = max(
        mapped["von_mises_top"], mapped["von_mises_bot"]
    )
    return mapped


def _patch_error_indicator(
    model: "FEModel",
    prepared: Mapping[int, Dict[str, Any]],
    nodal: Mapping[int, Mapping[str, float]],
) -> Dict[str, Any]:
    component_weights = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0] * 2, dtype=float)
    total_difference = 0.0
    total_recovered = 0.0
    per_element: Dict[int, float] = {}
    evaluated_points = 0
    for element_id in sorted(prepared):
        data = prepared[element_id]
        element = data["element"]
        if any(int(node_id) not in nodal for node_id in element.node_ids):
            continue
        nodal_values = np.array(
            [
                [float(nodal[int(node_id)][key]) for key in _SHELL_SURFACE_KEYS]
                for node_id in element.node_ids
            ],
            dtype=float,
        )
        coords = data["coords"]
        raw_values = data["values"]
        element_difference = 0.0
        element_recovered = 0.0
        for gp_index, ((xi, eta), gauss_weight) in enumerate(
            zip(np.asarray(element.gauss_points, dtype=float), np.asarray(element.gauss_weights, dtype=float))
        ):
            shape, dN_dxi, dN_deta = element.compute_shape_functions(float(xi), float(eta))
            recovered = np.asarray(shape, dtype=float) @ nodal_values
            difference = recovered - raw_values[gp_index]
            _rotation, _dN_dx, _dN_dy, det_j = element._local_frame_and_derivatives(
                coords, dN_dxi, dN_deta
            )
            weight = (
                abs(float(det_j))
                * abs(float(gauss_weight))
                * abs(float(element.thickness))
                * 0.5
            )
            element_difference += weight * float(
                np.dot(component_weights, difference**2)
            )
            element_recovered += weight * float(
                np.dot(component_weights, recovered**2)
            )
            evaluated_points += 1
        total_difference += element_difference
        total_recovered += element_recovered
        per_element[int(element_id)] = (
            float(np.sqrt(element_difference / element_recovered))
            if element_recovered > np.finfo(float).tiny
            else (0.0 if element_difference <= np.finfo(float).tiny else float("inf"))
        )
    relative = (
        float(np.sqrt(total_difference / total_recovered))
        if total_recovered > np.finfo(float).tiny
        else (0.0 if total_difference <= np.finfo(float).tiny else float("inf"))
    )
    return {
        "type": "normalized_global_surface_stress_l2",
        "is_energy_norm_estimate": False,
        "interpretation": (
            "diagnostic discrepancy between raw and recovered top/bottom "
            "surface stress tensors; not a ZZ compliance-energy error estimate"
        ),
        "status": "available" if evaluated_points else "unavailable",
        "relative": relative,
        "absolute_stress_norm": float(np.sqrt(max(total_difference, 0.0))),
        "recovered_stress_norm": float(np.sqrt(max(total_recovered, 0.0))),
        "evaluated_gauss_points": int(evaluated_points),
        "per_element_relative": per_element,
    }


def recover_shell_patch_stresses(
    model: "FEModel",
    element_stresses: Mapping[int, Mapping[str, Any]],
    *,
    element_ids: Optional[Sequence[int]] = None,
    config: Optional[PatchRecoveryConfig] = None,
    per_element_source: Optional[Mapping[int, str]] = None,
) -> Dict[str, Any]:
    """Recover continuous nodal shell surface stresses with guarded patches.

    A linear least-squares polynomial is fitted to the integration-point
    stresses in each node's incident-element patch.  The fit is accepted only
    for homogeneous full-integration Q4/Q8 topology or a mixed qualified
    E4-PL Q4/S3 patch, with matching material, thickness, provenance,
    planarity, normal orientation, full rank, and bounded condition number.
    Rank/conditioning failures fall back to formulation-native extrapolation
    and averaging *within the same continuity region*.  Legacy TRI3 remains
    unsupported.  Material, thickness, unqualified topology, provenance, or
    geometric discontinuities retain separate ``nodal_regions`` and never
    cross-average into a misleading single value.
    """

    selected_ids = (
        tuple(int(element_id) for element_id in element_ids)
        if element_ids is not None
        else tuple(int(element_id) for element_id in element_stresses)
    )
    require_model_element_capabilities(
        model,
        "patch_recovery",
        context="recover_shell_patch_stresses",
        element_ids=selected_ids,
    )
    settings = config if config is not None else PatchRecoveryConfig()
    sources = per_element_source or {}
    prepared, skipped, all_shell_incidence = _prepare_shell_patch_elements(
        model, element_stresses, element_ids, settings
    )
    incident_by_node: Dict[int, list[Tuple[int, Dict[str, Any]]]] = {}
    element_nodal = {
        int(element_id): data["element_nodal"]
        for element_id, data in prepared.items()
    }
    for element_id in sorted(prepared):
        data = prepared[element_id]
        for node_id in data["element"].node_ids:
            incident_by_node.setdefault(int(node_id), []).append((int(element_id), data))

    nodal: Dict[int, Dict[str, float]] = {}
    nodal_regions: Dict[int, list[Dict[str, Any]]] = {}
    diagnostics: Dict[int, Dict[str, Any]] = {}
    qualified: list[int] = []
    fallback: list[int] = []
    discontinuous: list[int] = []
    for node_id in sorted(all_shell_incidence):
        incident = sorted(
            incident_by_node.get(int(node_id), []),
            key=lambda item: item[0],
        )
        all_incident_ids = tuple(all_shell_incidence[int(node_id)])
        prepared_ids = {int(element_id) for element_id, _data in incident}
        missing_ids = [
            int(element_id)
            for element_id in all_incident_ids
            if int(element_id) not in prepared_ids
        ]
        if not incident:
            diagnostics[int(node_id)] = {
                "status": "skipped",
                "reason": "no_qualified_incident_shell",
                "incident_element_ids": list(all_incident_ids),
                "skipped_incident_element_ids": missing_ids,
            }
            discontinuous.append(int(node_id))
            continue

        combined_reason = _patch_guard_reason(incident, sources, settings)
        regions = _partition_patch_regions(incident, sources, settings)
        region_records: list[Dict[str, Any]] = []
        for region_index, region in enumerate(regions):
            node_coords = _patch_node_coordinates(node_id, region)
            values: Optional[np.ndarray] = None
            fit_diagnostics: Dict[str, Any] = {
                "sample_count": int(
                    sum(
                        data["positions"].shape[0]
                        for _element_id, data in region
                    )
                )
            }
            if node_coords is not None:
                values, fit_diagnostics = _fit_shell_patch(
                    node_coords, region, settings
                )
            if values is None:
                fallback_values = _patch_fallback_values(node_id, region)
                if fallback_values is not None:
                    values = fallback_values
                    fit_diagnostics.update(
                        status="fallback",
                        reason=str(
                            fit_diagnostics.get(
                                "reason", "patch_fit_failed"
                            )
                        ),
                    )
            record: Dict[str, Any] = {
                **fit_diagnostics,
                "region_index": int(region_index),
                "incident_element_ids": [
                    int(element_id) for element_id, _data in region
                ],
            }
            if values is not None:
                record["values"] = _nodal_surface_mapping(values)
            else:
                record.update(
                    status="skipped",
                    reason=str(
                        fit_diagnostics.get(
                            "reason", "no_region_fallback_values"
                        )
                    ),
                )
            region_records.append(record)
        nodal_regions[int(node_id)] = region_records

        complete_single_region = (
            len(region_records) == 1
            and not missing_ids
            and "values" in region_records[0]
        )
        if complete_single_region:
            record = region_records[0]
            nodal[int(node_id)] = dict(record["values"])
            diagnostics[int(node_id)] = {
                key: value
                for key, value in record.items()
                if key != "values"
            }
            if record.get("status") == "qualified":
                qualified.append(int(node_id))
            else:
                fallback.append(int(node_id))
            continue

        discontinuous.append(int(node_id))
        reason = combined_reason
        if reason is None and len(region_records) > 1:
            reason = "inconsistent_or_curved_shell_normals"
        if missing_ids:
            reason = "incomplete_patch_neighborhood"
        diagnostics[int(node_id)] = {
            "status": "discontinuous",
            "reason": reason or "multiple_continuity_regions",
            "incident_element_ids": list(all_incident_ids),
            "skipped_incident_element_ids": missing_ids,
            "region_count": len(region_records),
        }

    max_von_mises = 0.0
    max_von_mises_node: Optional[int] = None
    for node_id, records in sorted(nodal_regions.items()):
        for record in records:
            values = record.get("values")
            if not isinstance(values, Mapping):
                continue
            value = float(values["von_mises"])
            if value > max_von_mises:
                max_von_mises = value
                max_von_mises_node = int(node_id)
    error_indicator = (
        _patch_error_indicator(model, prepared, nodal)
        if settings.include_error_indicator
        else None
    )
    return {
        "method": (
            "zz_linear_patch_least_squares_with_"
            "continuity_preserving_fallback"
        ),
        "stress_frame": "global",
        "qualification": settings.to_dict(),
        "nodal": nodal,
        "nodal_regions": nodal_regions,
        "element_nodal": element_nodal,
        "node_diagnostics": diagnostics,
        "qualified_node_ids": qualified,
        "fallback_node_ids": fallback,
        "discontinuous_node_ids": discontinuous,
        "skipped_element_ids": sorted(skipped),
        "skipped_element_reasons": {
            int(element_id): reason for element_id, reason in sorted(skipped.items())
        },
        "max_von_mises": float(max_von_mises),
        "max_von_mises_node": max_von_mises_node,
        "error_indicator": error_indicator,
    }


def filter_reactions(
    reactions: Mapping[int, np.ndarray],
    recovery_config: Optional[RecoveryConfig] = None,
    model: Optional["FEModel"] = None,
) -> Dict[int, np.ndarray]:
    """Filter reaction dictionary by requested node ids."""

    recovery = default_recovery_config(recovery_config)
    if not recovery.include_reactions:
        return {}
    if recovery.node_ids is None:
        return {int(node_id): np.asarray(values, dtype=float) for node_id, values in reactions.items()}
    if model is not None:
        selected = set(recovery.selected_node_ids(model))
    else:
        selected = set(_optional_int_tuple(recovery.node_ids) or ())
    return {int(node_id): np.asarray(values, dtype=float) for node_id, values in reactions.items() if int(node_id) in selected}


def recovery_metadata(
    recovery_config: Optional[RecoveryConfig] = None,
    resource_config: Optional[ResourceConfig] = None,
    memory_estimate: Optional[MemoryEstimate] = None,
) -> Dict[str, Any]:
    """Serialize recovery/resource policy metadata for provenance records."""

    payload: Dict[str, Any] = {"recovery": default_recovery_config(recovery_config).to_dict()}
    if resource_config is not None:
        payload["resources"] = resource_config.to_dict()
    if memory_estimate is not None:
        payload["memory_estimate"] = memory_estimate.to_dict()
    return payload
