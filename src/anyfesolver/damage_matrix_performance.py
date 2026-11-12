"""Incremental CSR updates for impact damage stiffness and mass matrices."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Mapping, Sequence, Tuple

import numpy as np
from scipy import sparse

if TYPE_CHECKING:
    from .fe_core import FEModel


DamageElementTerm = Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


# The measured setup break-even is eleven *subsequent* matrix updates.  Keep
# this explicit and conservative: the event that triggers plan construction is
# not counted toward amortising the setup cost.
DAMAGE_MATRIX_PLAN_BREAK_EVEN_FUTURE_UPDATES = 11
DAMAGE_MATRIX_PLAN_MIN_OBSERVED_UPDATE_EVENTS = 2
DAMAGE_MATRIX_PLAN_DEFAULT_RETAINED_BYTES_CAP = 256 * 1024 * 1024


class DamageMatrixPlanFallback(ValueError):
    """Signal that a caller must use the exact scalar rebuild for this update."""


def estimate_damage_matrix_plan_retained_bytes(
    cached_terms: Tuple[int, Tuple[DamageElementTerm, ...]],
) -> int:
    """Return a conservative pre-build upper bound for plan-owned arrays.

    The bound assumes 64-bit CSR indices, no duplicate compression, and one
    possible point-mass diagonal entry per global degree of freedom.  It also
    includes the plan's base-value copies, per-element contribution maps, and
    a small object-overhead allowance.  This plan-only estimate excludes the
    cached legacy terms; the selection gate reports them separately and adds
    them because they remain alive for exact fallback throughout plan use.
    """

    total_dofs, terms = cached_terms
    total_dofs = max(int(total_dofs), 0)
    local_entries = sum(int(np.asarray(term[1]).size) for term in terms)
    element_count = len(terms)

    # K has at most ``local_entries`` stored positions; M can additionally
    # carry one point-mass entry per global degree of freedom.
    stiffness_nnz_upper = local_entries
    mass_nnz_upper = local_entries + total_dofs
    csr_index_bytes = np.dtype(np.intp).itemsize
    scalar_bytes = np.dtype(np.float64).itemsize
    csr_owned = (
        (stiffness_nnz_upper + mass_nnz_upper) * (scalar_bytes + csr_index_bytes)
        + 2 * (total_dofs + 1) * csr_index_bytes
    )
    base_values = (stiffness_nnz_upper + mass_nnz_upper) * scalar_bytes
    point_mass_maps = total_dofs * (csr_index_bytes + scalar_bytes)
    contribution_arrays = local_entries * (2 * csr_index_bytes + 2 * scalar_bytes)
    last_scales = element_count * scalar_bytes
    object_overhead = 4096 + 256 * element_count
    return int(
        csr_owned
        + base_values
        + point_mass_maps
        + contribution_arrays
        + last_scales
        + object_overhead
    )


def estimate_damage_matrix_cached_terms_retained_bytes(
    cached_terms: Tuple[int, Tuple[DamageElementTerm, ...]],
) -> int:
    """Return the array-buffer bytes retained by the exact fallback terms.

    The contact solvers intentionally keep these arrays alive after selecting
    an incremental plan so a revision or invalid input can fall back without
    changing exact scalar-rebuild semantics.  Their buffers therefore overlap
    the complete lifetime of a selected plan and must be charged to the same
    retained-memory allowance.
    """

    _total_dofs, terms = cached_terms
    return int(
        sum(
            int(np.asarray(array).nbytes)
            for term in terms
            for array in term[1:]
        )
    )


def estimate_damage_matrix_combined_retained_bytes(
    cached_terms: Tuple[int, Tuple[DamageElementTerm, ...]],
) -> int:
    """Return the peak retained bytes when the incremental plan is selected."""

    return int(
        estimate_damage_matrix_cached_terms_retained_bytes(cached_terms)
        + estimate_damage_matrix_plan_retained_bytes(cached_terms)
    )


@dataclass
class DamageMatrixPlanGate:
    """Conservative cost/memory selector for incremental damage matrices."""

    total_opportunities: int
    preflight_memory_bytes: int
    configured_memory_limit_bytes: int | None = None
    break_even_future_updates: int = DAMAGE_MATRIX_PLAN_BREAK_EVEN_FUTURE_UPDATES
    default_retained_bytes_cap: int = DAMAGE_MATRIX_PLAN_DEFAULT_RETAINED_BYTES_CAP
    min_observed_update_events: int = DAMAGE_MATRIX_PLAN_MIN_OBSERVED_UPDATE_EVENTS
    observed_update_events: int = 0
    observed_opportunities: int = 0
    remaining_opportunities: int = 0
    projected_future_update_events: int = 0
    cached_terms_retained_bytes: int = 0
    estimated_plan_retained_bytes: int = 0
    estimated_combined_retained_bytes: int = 0
    estimated_retained_bytes: int = 0
    actual_plan_retained_bytes: int = 0
    actual_combined_retained_bytes: int = 0
    accounted_combined_retained_bytes: int = 0
    actual_retained_bytes: int = 0
    legacy_update_count: int = 0
    plan_update_count: int = 0
    plan_build_count: int = 0
    rejected_plan_build_count: int = 0
    plan_update_fallback_count: int = 0
    cached_terms_refresh_count: int = 0
    model_revision_fallback_count: int = 0
    plan_selected: bool = False
    selection_reason: str = "no_damage_matrix_updates"
    _cached_terms_revision_key: Tuple[int, int, int, int] | None = field(default=None, repr=False)
    _disabled_reason: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.total_opportunities = max(int(self.total_opportunities), 0)
        self.preflight_memory_bytes = max(int(self.preflight_memory_bytes), 0)
        if self.configured_memory_limit_bytes is not None:
            self.configured_memory_limit_bytes = max(int(self.configured_memory_limit_bytes), 0)
        self.break_even_future_updates = max(int(self.break_even_future_updates), 1)
        self.default_retained_bytes_cap = max(int(self.default_retained_bytes_cap), 0)
        self.min_observed_update_events = max(int(self.min_observed_update_events), 1)

    @property
    def retained_memory_allowance_bytes(self) -> int:
        allowance = int(self.default_retained_bytes_cap)
        if self.configured_memory_limit_bytes is not None:
            configured_headroom = max(
                int(self.configured_memory_limit_bytes) - int(self.preflight_memory_bytes),
                0,
            )
            allowance = min(allowance, configured_headroom)
        return max(int(allowance), 0)

    def cached_terms_match(self, model: "FEModel") -> bool:
        return (
            self._cached_terms_revision_key is not None
            and self._cached_terms_revision_key == _revision_key(model)
        )

    @property
    def cached_terms_registered(self) -> bool:
        return self._cached_terms_revision_key is not None

    def register_cached_terms(self, model: "FEModel", *, replacing_stale: bool = False) -> None:
        if replacing_stale:
            self.cached_terms_refresh_count += 1
            self.model_revision_fallback_count += 1
            self.plan_selected = False
            self._disabled_reason = "model_revision_changed"
            self.selection_reason = "model_revision_changed_legacy_fallback"
        self._cached_terms_revision_key = _revision_key(model)

    def consider(
        self,
        cached_terms: Tuple[int, Tuple[DamageElementTerm, ...]],
        *,
        opportunity_index: int,
    ) -> bool:
        """Record an update event and decide whether setup can amortise."""

        self.observed_update_events += 1
        self.observed_opportunities = min(
            max(int(opportunity_index), 1),
            max(int(self.total_opportunities), 1),
        )
        self.remaining_opportunities = max(
            int(self.total_opportunities) - int(self.observed_opportunities),
            0,
        )
        observed_rate = min(
            float(self.observed_update_events) / float(self.observed_opportunities),
            1.0,
        )
        self.projected_future_update_events = int(
            np.floor(observed_rate * float(self.remaining_opportunities))
        )
        self.cached_terms_retained_bytes = (
            estimate_damage_matrix_cached_terms_retained_bytes(cached_terms)
        )
        self.estimated_plan_retained_bytes = (
            estimate_damage_matrix_plan_retained_bytes(cached_terms)
        )
        self.estimated_combined_retained_bytes = int(
            self.cached_terms_retained_bytes + self.estimated_plan_retained_bytes
        )
        self.accounted_combined_retained_bytes = self.estimated_combined_retained_bytes
        # Compatibility name: this is the complete incremental-selection
        # footprint now that the exact fallback terms are known to overlap the
        # plan lifetime.
        self.estimated_retained_bytes = self.estimated_combined_retained_bytes

        if self.plan_selected:
            self.selection_reason = "selected_cost_and_memory_gate_passed"
            return True
        if self._disabled_reason is not None:
            self.selection_reason = f"{self._disabled_reason}_legacy_fallback"
            return False
        if self.observed_update_events < self.min_observed_update_events:
            self.selection_reason = "insufficient_observed_update_events"
            return False
        if self.projected_future_update_events < self.break_even_future_updates:
            self.selection_reason = "projected_future_updates_below_break_even"
            return False
        if self.estimated_combined_retained_bytes > self.retained_memory_allowance_bytes:
            self.selection_reason = "estimated_retained_memory_exceeds_allowance"
            return False
        self.plan_selected = True
        self.selection_reason = "selected_cost_and_memory_gate_passed"
        return True

    def accept_built_plan(self, plan: "DamageMatrixPlan") -> bool:
        self.plan_build_count += 1
        self.actual_plan_retained_bytes = max(int(plan.retained_bytes), 0)
        self.actual_combined_retained_bytes = int(
            self.cached_terms_retained_bytes + self.actual_plan_retained_bytes
        )
        # Continue to charge the conservative pre-build plan allowance after
        # construction: ``plan.retained_bytes`` counts ndarray/CSR buffers but
        # intentionally omits the plan's Python maps and object overhead.
        self.accounted_combined_retained_bytes = max(
            int(self.estimated_combined_retained_bytes),
            int(self.actual_combined_retained_bytes),
        )
        self.actual_retained_bytes = self.actual_combined_retained_bytes
        if self.accounted_combined_retained_bytes > self.retained_memory_allowance_bytes:
            self.rejected_plan_build_count += 1
            self.plan_selected = False
            self._disabled_reason = "actual_retained_memory_exceeds_allowance"
            self.selection_reason = "actual_retained_memory_exceeds_allowance_legacy_fallback"
            return False
        return True

    def reject_plan_setup(self, reason: str) -> None:
        self.rejected_plan_build_count += 1
        self.plan_selected = False
        self._disabled_reason = str(reason)
        self.selection_reason = f"{self._disabled_reason}_legacy_fallback"

    def record_plan_update_fallback(self, reason: str) -> None:
        self.plan_update_fallback_count += 1
        self.plan_selected = False
        self._disabled_reason = str(reason)
        self.selection_reason = f"{self._disabled_reason}_legacy_fallback"

    def record_update(self, *, used_plan: bool) -> None:
        if used_plan:
            self.plan_update_count += 1
        else:
            self.legacy_update_count += 1

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "fast_path_name": "incremental_damage_csr_updates",
            "setup_cost_model": "require_benchmarked_break_even_in_future_updates",
            "event_opportunity_model": (
                "floor(observed update density * remaining nominal time steps)"
            ),
            "retained_memory_model": (
                "simultaneously retained cached legacy term arrays plus conservative "
                "plan-owned arrays, maps, and object overhead"
            ),
            "plan_selected": bool(self.plan_selected),
            "selection_reason": str(self.selection_reason),
            "break_even_future_update_events": int(self.break_even_future_updates),
            "min_observed_update_events": int(self.min_observed_update_events),
            "observed_update_events": int(self.observed_update_events),
            "observed_opportunities": int(self.observed_opportunities),
            "remaining_opportunities": int(self.remaining_opportunities),
            "projected_future_update_events": int(self.projected_future_update_events),
            "default_retained_memory_cap_bytes": int(self.default_retained_bytes_cap),
            "configured_memory_limit_bytes": (
                None
                if self.configured_memory_limit_bytes is None
                else int(self.configured_memory_limit_bytes)
            ),
            "preflight_memory_bytes": int(self.preflight_memory_bytes),
            "retained_memory_allowance_bytes": int(self.retained_memory_allowance_bytes),
            "cached_terms_retained_bytes": int(self.cached_terms_retained_bytes),
            "estimated_plan_retained_bytes": int(self.estimated_plan_retained_bytes),
            "estimated_combined_retained_bytes": int(self.estimated_combined_retained_bytes),
            "estimated_retained_bytes": int(self.estimated_retained_bytes),
            "actual_plan_retained_bytes": int(self.actual_plan_retained_bytes),
            "actual_combined_retained_bytes": int(self.actual_combined_retained_bytes),
            "accounted_combined_retained_bytes": int(self.accounted_combined_retained_bytes),
            "actual_retained_bytes": int(self.actual_retained_bytes),
            "legacy_update_count": int(self.legacy_update_count),
            "plan_update_count": int(self.plan_update_count),
            "plan_build_count": int(self.plan_build_count),
            "rejected_plan_build_count": int(self.rejected_plan_build_count),
            "plan_update_fallback_count": int(self.plan_update_fallback_count),
            "cached_terms_refresh_count": int(self.cached_terms_refresh_count),
            "model_revision_fallback_count": int(self.model_revision_fallback_count),
        }


def _revision_key(model: "FEModel") -> Tuple[int, int, int, int]:
    revisions = model.mesh.revision_signature()
    return (
        int(revisions.get("topology", 0)),
        int(revisions.get("geometry", 0)),
        int(revisions.get("material", 0)),
        int(revisions.get("mass", 0)),
    )


def _point_mass_diagonal(model: "FEModel", total_dofs: int) -> np.ndarray:
    diagonal = np.zeros(total_dofs, dtype=float)
    for node_id, mass in (getattr(model.mesh, "point_masses", None) or {}).items():
        node = model.mesh.get_node(int(node_id))
        if node is None or float(mass) == 0.0:
            continue
        for axis in range(3):
            diagonal[node.dofs[axis]] += float(mass)
    return diagonal


def _csr_position_map(matrix: sparse.csr_matrix) -> Tuple[Dict[int, int], ...]:
    rows = []
    for row in range(matrix.shape[0]):
        start = int(matrix.indptr[row])
        stop = int(matrix.indptr[row + 1])
        rows.append(
            {
                int(column): position
                for position, column in zip(
                    range(start, stop),
                    matrix.indices[start:stop],
                )
            }
        )
    return tuple(rows)


def _positions(
    row_maps: Sequence[Mapping[int, int]],
    rows: np.ndarray,
    columns: np.ndarray,
) -> np.ndarray:
    return np.fromiter(
        (
            row_maps[int(row)][int(column)]
            for row, column in zip(rows, columns)
        ),
        dtype=np.intp,
        count=int(rows.size),
    )


@dataclass(frozen=True)
class DamageContribution:
    element_id: int
    stiffness_positions: np.ndarray
    mass_positions: np.ndarray
    stiffness_values: np.ndarray
    mass_values: np.ndarray


@dataclass
class DamageMatrixPlan:
    """One bounded matrix pattern with per-element additive contribution maps."""

    revision_key: Tuple[int, int, int, int]
    stiffness: sparse.csr_matrix
    mass: sparse.csr_matrix
    contributions: Tuple[DamageContribution, ...]
    base_stiffness_data: np.ndarray
    base_mass_data: np.ndarray
    point_mass_positions: np.ndarray
    point_mass_values: np.ndarray
    setup_seconds: float
    retained_bytes: int
    last_scales: np.ndarray
    update_count: int = 0
    no_change_count: int = 0
    changed_element_count: int = 0
    update_seconds: float = 0.0
    invalidation_count: int = 0
    fallback_count: int = 0
    fallback_reason: str | None = None
    _id_to_index: Dict[int, int] = field(default_factory=dict, repr=False)
    _active_indices: set[int] = field(default_factory=set, repr=False)

    @classmethod
    def build(
        cls,
        model: "FEModel",
        cached_terms: Tuple[int, Tuple[DamageElementTerm, ...]],
    ) -> "DamageMatrixPlan":
        start = time.perf_counter()
        total_dofs, terms = cached_terms
        if not terms:
            empty_k = sparse.csr_matrix((total_dofs, total_dofs), dtype=float)
            empty_m = sparse.csr_matrix((total_dofs, total_dofs), dtype=float)
            return cls(
                revision_key=_revision_key(model),
                stiffness=empty_k,
                mass=empty_m,
                contributions=(),
                base_stiffness_data=np.empty(0, dtype=float),
                base_mass_data=np.empty(0, dtype=float),
                point_mass_positions=np.empty(0, dtype=np.intp),
                point_mass_values=np.empty(0, dtype=float),
                setup_seconds=float(time.perf_counter() - start),
                retained_bytes=0,
                last_scales=np.empty(0, dtype=float),
            )

        rows = np.concatenate([term[1] for term in terms])
        columns = np.concatenate([term[2] for term in terms])
        stiffness_values = np.concatenate([term[3] for term in terms])
        mass_values = np.concatenate([term[4] for term in terms])
        stiffness = sparse.coo_matrix(
            (stiffness_values, (rows, columns)),
            shape=(total_dofs, total_dofs),
        ).tocsr()
        # Include point-mass diagonals in the initial COO conversion.  Sparse
        # matrix addition drops explicit zero entries; doing ``M + diag`` here
        # would therefore destroy positions needed by zero-valued element
        # contributions and make later in-place updates unsafe.
        point_mass = _point_mass_diagonal(model, total_dofs)
        point_mass_dofs = np.flatnonzero(point_mass).astype(np.intp, copy=False)
        mass_coo_rows = rows
        mass_coo_columns = columns
        mass_coo_data = mass_values
        if point_mass_dofs.size:
            mass_coo_rows = np.concatenate((mass_coo_rows, point_mass_dofs))
            mass_coo_columns = np.concatenate((mass_coo_columns, point_mass_dofs))
            mass_coo_data = np.concatenate((mass_coo_data, point_mass[point_mass_dofs]))
        mass = sparse.coo_matrix(
            (mass_coo_data, (mass_coo_rows, mass_coo_columns)),
            shape=(total_dofs, total_dofs),
        ).tocsr()
        stiffness.sum_duplicates()
        stiffness.sort_indices()
        mass.sum_duplicates()
        mass.sort_indices()
        stiffness_rows = _csr_position_map(stiffness)
        mass_rows = _csr_position_map(mass)
        point_mass_positions = _positions(mass_rows, point_mass_dofs, point_mass_dofs)
        point_mass_values = point_mass[point_mass_dofs].copy()
        base_stiffness_data = stiffness.data.copy()
        base_mass_data = mass.data.copy()
        contributions = []
        retained_bytes = int(
            stiffness.data.nbytes
            + stiffness.indices.nbytes
            + stiffness.indptr.nbytes
            + mass.data.nbytes
            + mass.indices.nbytes
            + mass.indptr.nbytes
            + base_stiffness_data.nbytes
            + base_mass_data.nbytes
            + point_mass_positions.nbytes
            + point_mass_values.nbytes
        )
        for element_id, row, column, k_flat, m_flat in terms:
            k_positions = _positions(stiffness_rows, row, column)
            m_positions = _positions(mass_rows, row, column)
            k_values = np.asarray(k_flat, dtype=float).copy()
            m_values = np.asarray(m_flat, dtype=float).copy()
            contributions.append(
                DamageContribution(
                    element_id=int(element_id),
                    stiffness_positions=k_positions,
                    mass_positions=m_positions,
                    stiffness_values=k_values,
                    mass_values=m_values,
                )
            )
            retained_bytes += int(
                k_positions.nbytes
                + m_positions.nbytes
                + k_values.nbytes
                + m_values.nbytes
            )
        contribution_tuple = tuple(contributions)
        last_scales = np.ones(len(contribution_tuple), dtype=float)
        retained_bytes += int(last_scales.nbytes)
        return cls(
            revision_key=_revision_key(model),
            stiffness=stiffness,
            mass=mass,
            contributions=contribution_tuple,
            base_stiffness_data=base_stiffness_data,
            base_mass_data=base_mass_data,
            point_mass_positions=point_mass_positions,
            point_mass_values=point_mass_values,
            setup_seconds=float(time.perf_counter() - start),
            retained_bytes=retained_bytes,
            last_scales=last_scales,
            _id_to_index={
                contribution.element_id: index
                for index, contribution in enumerate(contribution_tuple)
            },
        )

    def is_valid(self, model: "FEModel") -> bool:
        return self.revision_key == _revision_key(model)

    def update(
        self,
        model: "FEModel",
        element_scales: Mapping[int, float],
    ) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
        if not self.is_valid(model):
            self.invalidation_count += 1
            self.fallback_count += 1
            self.fallback_reason = "model_revision_changed"
            raise DamageMatrixPlanFallback("damage matrix plan is invalid after a model revision")
        start = time.perf_counter()
        raw_scales = {int(element_id): float(scale) for element_id, scale in element_scales.items()}
        if any(not np.isfinite(scale) for scale in raw_scales.values()):
            self.fallback_count += 1
            self.fallback_reason = "nonfinite_element_scale"
            raise DamageMatrixPlanFallback("incremental damage scales must be finite")
        scales = {
            element_id: min(max(scale, 0.0), 1.0)
            for element_id, scale in raw_scales.items()
        }
        candidate_indices = set(self._active_indices)
        candidate_indices.update(
            self._id_to_index[element_id]
            for element_id in scales
            if element_id in self._id_to_index
        )
        changed = 0
        for index in sorted(candidate_indices):
            contribution = self.contributions[index]
            new_scale = float(scales.get(contribution.element_id, 1.0))
            delta = new_scale - float(self.last_scales[index])
            if delta == 0.0:
                continue
            np.add.at(
                self.stiffness.data,
                contribution.stiffness_positions,
                delta * contribution.stiffness_values,
            )
            np.add.at(
                self.mass.data,
                contribution.mass_positions,
                delta * contribution.mass_values,
            )
            self.last_scales[index] = new_scale
            if new_scale == 1.0:
                self._active_indices.discard(index)
            else:
                self._active_indices.add(index)
            changed += 1
        if changed and not self._active_indices:
            # Resetting every omitted/explicit scale to one must not retain
            # roundoff accumulated by a long damage history.
            np.copyto(self.stiffness.data, self.base_stiffness_data)
            np.copyto(self.mass.data, self.base_mass_data)
        elif (
            changed
            and len(self._active_indices) == len(self.contributions)
            and bool(np.all(self.last_scales == 0.0))
        ):
            # The legacy scalar assembly multiplies every local value by
            # exactly zero.  Restore that exact result (plus point masses)
            # instead of leaving subtraction roundoff in shared CSR entries.
            self.stiffness.data.fill(0.0)
            self.mass.data.fill(0.0)
            self.mass.data[self.point_mass_positions] = self.point_mass_values
        self.update_count += 1
        self.changed_element_count += changed
        if changed == 0:
            self.no_change_count += 1
        self.update_seconds += float(time.perf_counter() - start)
        return self.stiffness, self.mass

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "fast_path_name": "incremental_damage_csr_updates",
            "eligible_element_count": len(self.contributions),
            "fallback_element_count": 0,
            "fallback_reason": self.fallback_reason,
            "revision_key": list(self.revision_key),
            "setup_seconds": float(self.setup_seconds),
            "retained_bytes": int(self.retained_bytes),
            "update_count": int(self.update_count),
            "no_change_count": int(self.no_change_count),
            "changed_element_count": int(self.changed_element_count),
            "update_seconds": float(self.update_seconds),
            "invalidation_count": int(self.invalidation_count),
            "fallback_count": int(self.fallback_count),
            "active_scaled_element_count": len(self._active_indices),
            "stiffness_nnz": int(self.stiffness.nnz),
            "mass_nnz": int(self.mass.nnz),
        }
