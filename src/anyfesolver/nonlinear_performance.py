"""Performance layer for nonlinear shell/beam analyses.

The production nonlinear formulations remain in :mod:`anyfesolver.nonlinear_static`
and the element modules.  This module removes repeated Python and sparse-matrix
bookkeeping from every Newton iteration by building a persistent assembly plan
for each model/layer-count pair.

The plan caches:

* shell groups and reference-geometry arrays,
* element DOF mappings,
* reusable shell input work arrays,
* global force scatter positions,
* the unique global CSR tangent pattern, and
* local-entry-to-CSR-data scatter positions.

It also installs two low-risk solver improvements:

* revision-counter sparsity caching instead of rebuilding a JSON/SHA topology
  signature on every cache lookup;
* displacement-control block elimination using two right-hand sides on the
  ordinary structural tangent instead of an augmented sparse matrix.

The installation is idempotent.  The nonlinear solver activates it lazily on
first use.  Set ``FE_SOLVER_DISABLE_FAST_NL=1`` before the first nonlinear
solve to retain the legacy assembly path for comparison and debugging.
"""

from __future__ import annotations

import copy
import os
import threading
import time
import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

from .jit_compiler import njit
from . import _q4_assumed_strain_plasticity as _q4_assumed_strain
from .initial_field_state import state_has_active_initial_fields
from .nonlinear_analysis_diagnostics import record_nonlinear_assembly_execution
from .nonlinear_element_evaluation import (
    evaluate_nonlinear_element,
    require_legacy_direct_nonlinear_element,
)
from .nonlinear_state import (
    NonlinearStateStore,
    PersistentStateEligibilityError,
    ShellStateBatch,
    ShellStateLayout,
    StateTrialToken,
    begin_state_evaluation,
    discard_active_state_candidate,
    finish_state_evaluation,
)


# Batch B enables this only while its generalized elastic kernel is installed.
# Keeping the flag here prevents the legacy/vectorized constitutive kernel from
# receiving orthotropic or pre-integrated sections when Numba is unavailable.
_EXTENDED_ELASTIC_S4_BATCHES = False

# Batch kind of qualified plastic Q4 shells on the assumed-strain route.
_Q4_ASSUMED_STRAIN_SHELL_KIND = "isotropic_q4_assumed_strain_plastic"

if TYPE_CHECKING:
    from .fe_core import FEModel, FEMesh


_INITIAL_FIELD_KEYS = (
    "initial_membrane_stress",
    "initial_bending_stress",
    "initial_membrane_prestrain",
    "initial_curvature_prestrain",
    "initial_fiber_stress",
    "initial_fiber_prestrain",
)


def _state_has_initial_fields(state: Any) -> bool:
    return state_has_active_initial_fields(state, _INITIAL_FIELD_KEYS)


def _apply_initial_field_shell_overrides(
    plan: "NonlinearAssemblyPlan",
    batch: "_ShellBatchPlan",
    displacements: np.ndarray,
    committed_states: Mapping[int, Any],
    tangent: bool,
    trial_states: Dict[int, Any],
    *,
    deleted_element_ids: Sequence[int] = (),
    residual_stiffness_fraction: float = 1.0,
) -> int:
    """Replace only initialized shell entries with the scalar exact response.

    The compiled ordinary-shell kernel intentionally has no immutable initial
    field inputs.  Evaluating the full batch and replacing the uncommon
    initialized members preserves the fast path for every unaffected shell in
    the same group without changing the element state contract.
    """

    model = plan.model
    displacement_array = np.asarray(displacements, dtype=float)
    deleted = {int(element_id) for element_id in deleted_element_ids}
    residual_fraction = float(residual_stiffness_fraction)
    override_count = 0
    for index, (element_id, element) in enumerate(zip(batch.element_ids, batch.elements)):
        element_key = int(element_id)
        state = committed_states.get(element_key)
        if not _state_has_initial_fields(state):
            continue
        if getattr(batch, "_batch_b_initial_fields_supported", False):
            continue
        material = model.get_material(element.material_name)
        require_legacy_direct_nonlinear_element(
            element,
            context="compiled shell initial-field override",
        )
        force, stiffness, trial_state = element.compute_nonlinear_response(
            model.mesh,
            material,
            displacement_array[batch.dof_mappings[index]],
            state,
            plan.num_layers,
            tangent,
        )
        if element_key in deleted:
            # The vectorized batch has already applied deletion scaling, but
            # this exact initial-field replacement is evaluated independently.
            # Apply the same residual policy here and discard its constitutive
            # candidate so deletion-time history remains frozen.
            force = residual_fraction * np.asarray(force, dtype=float)
            if tangent and stiffness is not None:
                stiffness = residual_fraction * np.asarray(
                    stiffness, dtype=float
                )
        plan.force_values[batch.force_positions[index]] = np.asarray(force, dtype=float).reshape(-1)
        if tangent and stiffness is not None:
            plan.tangent_values[batch.tangent_positions[index]] = np.asarray(
                stiffness, dtype=float
            ).reshape(-1)
        if element_key in deleted and state is not None:
            trial_states[element_key] = state
        elif trial_state is not None:
            trial_states[element_key] = trial_state
        override_count += 1
    return override_count


@njit(cache=True)
def _scatter_sum(values: np.ndarray, positions: np.ndarray, output_size: int) -> np.ndarray:
    """Sum flat local values into a precomputed unique global index space."""
    result = np.zeros(output_size, dtype=np.float64)
    for index in range(values.size):
        result[positions[index]] += values[index]
    return result


def _scatter_sum_python_ordered(
    values: np.ndarray, positions: np.ndarray, output_size: int
) -> np.ndarray:
    """Unbuffered binary64 additions in original order for owned flat buffers.

    ``add.at`` handles duplicate indices without regrouping their additions.
    Other layouts retain the original kernel, including its error behavior.
    This helper is selected only for exact generic Beam plans with JIT disabled.
    """
    if (
        type(values) is not np.ndarray
        or values.dtype != np.dtype(np.float64)
        or values.ndim != 1
        or type(positions) is not np.ndarray
        or positions.ndim != 1
        or positions.dtype != np.dtype(np.intp)
        or positions.shape != values.shape
        or type(output_size) is not int
        or output_size < 0
    ):
        return _scatter_sum(values, positions, output_size)
    result = np.zeros(output_size, dtype=np.float64)
    np.add.at(result, positions, values)
    return result


def _generic_beam_scatter_kernel(plan: "NonlinearAssemblyPlan") -> Any:
    """Select a layout-only optimization, without caching model eligibility."""
    import sys
    from .elements import BeamElement

    numba = sys.modules.get("numba")
    if (
        bool(getattr(getattr(numba, "config", None), "DISABLE_JIT", False))
        and not plan.shell_batches
        and not plan.quadratic_beam_batches
        and plan.non_shell_elements
        and all(type(record.element) is BeamElement for record in plan.non_shell_elements)
    ):
        return _scatter_sum_python_ordered
    return _scatter_sum


def _revision_tuple(mesh: "FEMesh") -> Tuple[int, ...]:
    revisions = getattr(mesh, "revision_signature", lambda: {})()
    return tuple(
        int(revisions.get(name, 0))
        for name in ("topology", "geometry", "material", "mpc")
    )


def _sparsity_revision_tuple(mesh: "FEMesh") -> Tuple[int, int, int]:
    revisions = getattr(mesh, "revision_signature", lambda: {})()
    direct_token = getattr(mesh, "_qualified_direct_state_token", None)
    direct_revision = (
        int(direct_token[0])
        if isinstance(direct_token, list) and len(direct_token) == 1
        else -1
    )
    return (
        int(revisions.get("topology", 0)),
        int(revisions.get("mpc", 0)),
        direct_revision,
    )


def _qualified_plan_input_signature(model: "FEModel") -> Tuple[Any, ...]:
    """Bind exact mutable inputs snapshotted by qualified shell batches."""

    from .e4_pl_s3_state import canonical_json_bytes, resolved_material_descriptor
    from .elements import _generalized_shell_section_cache_fingerprint

    mesh = model.mesh
    token = getattr(mesh, "_qualified_direct_state_token", None)
    direct_revision = (
        int(token[0])
        if isinstance(token, list) and len(token) == 1
        else -1
    )
    records: List[Tuple[Any, ...]] = []
    materials: Dict[int, Tuple[Any, ...]] = {}
    sections: Dict[int, Tuple[Any, ...]] = {}
    validated_families: set[str] = set()
    for raw_element_id, element in sorted(mesh.elements.items()):
        formulation_id = str(getattr(element, "formulation_id", ""))
        if formulation_id not in {
            "E4_PL_QUALIFIED_Q4_HYBRID_V2",
            "E4_PL_QUALIFIED_S3_COMPANION_V1",
        }:
            continue
        if formulation_id not in validated_families:
            validator = getattr(element, "validate_quadrature_authority", None)
            if not callable(validator):
                raise ValueError(
                    "qualified nonlinear plan lacks exact quadrature authority"
                )
            validator()
            validated_families.add(formulation_id)
        material = model.get_material(element.material_name)
        material_identity = id(material)
        if material_identity not in materials:
            materials[material_identity] = (
                type(material).__module__,
                type(material).__qualname__,
                canonical_json_bytes(resolved_material_descriptor(material)),
            )
        section = getattr(element, "shell_section", None)
        section_identity = id(section) if section is not None else 0
        if section is not None and section_identity not in sections:
            sections[section_identity] = (
                _generalized_shell_section_cache_fingerprint(section)
            )
        records.append(
            (
                int(raw_element_id),
                type(element).__module__,
                type(element).__qualname__,
                formulation_id,
                tuple(int(value) for value in element.node_ids),
                str(element.material_name),
                material_identity,
                section_identity,
                int(element.__dict__.get("_qualified_plan_state_revision", -1)),
            )
        )
    return (
        direct_revision,
        tuple(records),
        tuple(sorted(materials.items())),
        tuple(sorted(sections.items())),
    )


@dataclass
class NonlinearAssemblyTimings:
    calls: int = 0
    tangent_calls: int = 0
    residual_only_calls: int = 0
    shell_kernel_seconds: float = 0.0
    non_shell_seconds: float = 0.0
    state_pack_seconds: float = 0.0
    force_scatter_seconds: float = 0.0
    tangent_scatter_seconds: float = 0.0
    total_seconds: float = 0.0
    initial_field_override_seconds: float = 0.0
    initial_field_override_elements: int = 0
    initial_field_accelerated_elements: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calls": int(self.calls),
            "tangent_calls": int(self.tangent_calls),
            "residual_only_calls": int(self.residual_only_calls),
            "shell_kernel_seconds": float(self.shell_kernel_seconds),
            "non_shell_seconds": float(self.non_shell_seconds),
            "state_pack_seconds": float(self.state_pack_seconds),
            "force_scatter_seconds": float(self.force_scatter_seconds),
            "tangent_scatter_seconds": float(self.tangent_scatter_seconds),
            "total_seconds": float(self.total_seconds),
            "initial_field_override_seconds": float(self.initial_field_override_seconds),
            "initial_field_override_elements": int(self.initial_field_override_elements),
            "initial_field_accelerated_elements": int(self.initial_field_accelerated_elements),
        }


@dataclass
class _ElementScatter:
    element_id: int
    element: Any
    dof_mapping: np.ndarray
    force_positions: np.ndarray
    tangent_positions: np.ndarray


@dataclass
class _ShellBatchPlan:
    key: Tuple[Any, ...]
    element_ids: np.ndarray
    elements: Tuple[Any, ...]
    dof_mappings: np.ndarray
    force_positions: np.ndarray
    tangent_positions: np.ndarray
    T0: np.ndarray
    B_m: np.ndarray
    B_b: np.ndarray
    B_d: np.ndarray
    Gw: np.ndarray
    detw: np.ndarray
    B_s: np.ndarray
    detw_shear: np.ndarray
    linear_corrections: np.ndarray
    material: Any
    thickness: float
    drilling_stabilization: float
    n_gp: int
    n_dof: int
    num_layers: int
    state_layout: ShellStateLayout
    u_work: np.ndarray
    plastic_work: np.ndarray
    alpha_work: np.ndarray

    @property
    def has_plasticity(self) -> bool:
        return getattr(self.material, "hardening_curve", None) is not None

    def _shell_kernel(self) -> Any:
        """Return the batch kernel for this plan's constitutive route.

        Qualified plastic Q4 batches evaluate the layered return map at the
        hybrid assumed strain (:mod:`anyfesolver._q4_assumed_strain_plasticity`);
        every other batch keeps the ordinary compatible kernel.
        """

        from .vectorized_nonlinear import batch_shell_nonlinear_response

        if not self.key or self.key[0] != _Q4_ASSUMED_STRAIN_SHELL_KIND:
            return batch_shell_nonlinear_response
        offsets = self.__dict__.get("_q4_assumed_strain_offsets")
        if offsets is None:
            raise RuntimeError("qualified Q4 assumed-strain batch lacks its offsets")

        def kernel(*arguments: Any) -> Any:
            return _q4_assumed_strain.kernel_batch(*arguments, offsets)[:5]

        return kernel

    @classmethod
    def build(
        cls,
        model: "FEModel",
        key: Tuple[Any, ...],
        items: Sequence[Tuple[int, Any, _ElementScatter]],
        num_layers: int,
    ) -> "_ShellBatchPlan":
        first = items[0][1]
        first_cache = first._nonlinear_geometry(model.mesh)
        n_elem = len(items)
        n_dof = int(first.total_dofs)
        n_gp = int(first_cache["detw_all"].shape[0])
        n_shear = int(first_cache["detw_shear_all"].shape[0])

        dof_mappings = np.empty((n_elem, n_dof), dtype=np.intp)
        force_positions = np.empty((n_elem, n_dof), dtype=np.intp)
        tangent_positions = np.empty((n_elem, n_dof * n_dof), dtype=np.intp)
        T0 = np.empty((n_elem, n_dof, n_dof), dtype=float)
        B_m = np.empty((n_elem, n_gp, 3, n_dof), dtype=float)
        B_b = np.empty((n_elem, n_gp, 3, n_dof), dtype=float)
        B_d = np.empty((n_elem, n_gp, 1, n_dof), dtype=float)
        Gw = np.empty((n_elem, n_gp, 2, n_dof), dtype=float)
        detw = np.empty((n_elem, n_gp), dtype=float)
        B_s = np.empty((n_elem, n_shear, 2, n_dof), dtype=float)
        detw_shear = np.empty((n_elem, n_shear), dtype=float)
        element_ids = np.empty(n_elem, dtype=np.int64)
        elements: List[Any] = []
        linear_corrections = np.zeros((n_elem, n_dof, n_dof), dtype=float)
        correction_cache: Dict[Tuple[Any, ...], np.ndarray] = {}

        for batch_index, (element_id, element, scatter) in enumerate(items):
            cache = element._nonlinear_geometry(model.mesh)
            element_ids[batch_index] = int(element_id)
            elements.append(element)
            dof_mappings[batch_index] = scatter.dof_mapping
            force_positions[batch_index] = scatter.force_positions
            tangent_positions[batch_index] = scatter.tangent_positions
            T0[batch_index] = cache["T0"]
            B_m[batch_index] = cache["B_m_all"]
            B_b[batch_index] = cache["B_b_all"]
            B_d[batch_index] = cache["B_d_all"]
            Gw[batch_index] = cache["Gw_all"]
            detw[batch_index] = cache["detw_all"]
            B_s[batch_index] = cache["B_s_all"]
            detw_shear[batch_index] = cache["detw_shear_all"]
            correction_builder = getattr(element, "_qualified_linear_correction", None)
            if callable(correction_builder):
                material = model.get_material(element.material_name)
                cache_key_builder = getattr(
                    element, "_qualified_stiffness_cache_key", None
                )
                correction_key = (
                    int(num_layers),
                    cache_key_builder(model.mesh, material)
                    if callable(cache_key_builder)
                    else (int(element_id),),
                )
                correction = correction_cache.get(correction_key)
                if correction is None:
                    correction = np.asarray(
                        correction_builder(model.mesh, material, int(num_layers)),
                        dtype=float,
                    )
                    correction_cache[correction_key] = correction
                linear_corrections[batch_index] = correction

        material = model.get_material(first.material_name)
        plan = cls(
            key=key,
            element_ids=element_ids,
            elements=tuple(elements),
            dof_mappings=dof_mappings,
            force_positions=force_positions,
            tangent_positions=tangent_positions,
            T0=T0,
            B_m=B_m,
            B_b=B_b,
            B_d=B_d,
            Gw=Gw,
            detw=detw,
            B_s=B_s,
            detw_shear=detw_shear,
            linear_corrections=linear_corrections,
            material=material,
            thickness=float(first.thickness),
            drilling_stabilization=float(first.drilling_stabilization),
            n_gp=n_gp,
            n_dof=n_dof,
            num_layers=int(num_layers),
            state_layout=ShellStateLayout.from_dimensions(
                element_ids,
                n_gp,
                int(num_layers),
            ),
            u_work=np.zeros((n_elem, n_dof), dtype=float),
            plastic_work=np.zeros((n_elem, n_gp * num_layers, 3), dtype=float),
            alpha_work=np.zeros((n_elem, n_gp * num_layers), dtype=float),
        )
        if key and key[0] == _Q4_ASSUMED_STRAIN_SHELL_KIND:
            offsets = np.asarray([
                _q4_assumed_strain.offset_operator(element, model.mesh, material)
                for element in elements
            ])
            offsets.setflags(write=False)
            plan.__dict__["_q4_assumed_strain_offsets"] = offsets
        return plan

    def evaluate(
        self,
        displacements: np.ndarray,
        committed_states: Mapping[int, Any],
        tangent: bool,
        deleted_element_ids: Sequence[int] = (),
        residual_stiffness_fraction: float = 1.0,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[int, Any], float]:
        from .vectorized_nonlinear import batch_shell_nonlinear_response

        start = time.perf_counter()
        deleted = {int(element_id) for element_id in deleted_element_ids}
        residual_fraction = float(residual_stiffness_fraction)
        self.u_work[:] = np.asarray(displacements, dtype=float)[self.dof_mappings]

        if self.has_plasticity:
            self.plastic_work.fill(0.0)
            self.alpha_work.fill(0.0)
            for index, (element_id, element) in enumerate(zip(self.element_ids, self.elements)):
                state = committed_states.get(int(element_id))
                if state is None:
                    state = element.init_nonlinear_state(self.num_layers)
                self.plastic_work[index] = np.asarray(state["plastic_strain"], dtype=float)
                self.alpha_work[index] = np.asarray(state["alpha"], dtype=float)

        F_batch, K_batch, ep_new, alpha_new, layer_strain = self._shell_kernel()(
            self.u_work,
            self.T0,
            self.B_m,
            self.B_b,
            self.B_d,
            self.Gw,
            self.detw,
            self.B_s,
            self.detw_shear,
            float(self.material.elastic_modulus),
            float(self.material.poisson_ratio),
            float(self.material.shear_modulus),
            self.thickness,
            self.drilling_stabilization,
            bool(tangent),
            getattr(self.material, "hardening_curve", None),
            self.plastic_work,
            self.alpha_work,
            self.num_layers,
        )
        if np.any(self.linear_corrections):
            F_batch += np.einsum(
                "eij,ej->ei",
                self.linear_corrections,
                self.u_work,
                optimize=True,
            )
            if tangent and K_batch is not None:
                K_batch += self.linear_corrections

        states: Dict[int, Any] = {}
        if self.has_plasticity:
            points_per_element = self.n_gp * self.num_layers
            for index, (element_id, element) in enumerate(
                zip(self.element_ids, self.elements)
            ):
                start_layer = index * points_per_element
                stop_layer = start_layer + points_per_element
                if int(element_id) in deleted:
                    existing = committed_states.get(int(element_id))
                    if isinstance(existing, dict):
                        states[int(element_id)] = existing
                    else:
                        states[int(element_id)] = {
                            "plastic_strain": ep_new[index],
                            "alpha": alpha_new[index],
                            "layer_strain": layer_strain[start_layer:stop_layer].copy(),
                        }
                    continue
                state = {
                    "plastic_strain": ep_new[index],
                    "alpha": alpha_new[index],
                    "layer_strain": layer_strain[start_layer:stop_layer].copy(),
                }
                attach_algorithmic_origin = getattr(
                    element,
                    "attach_current_tangent_algorithmic_origin",
                    None,
                )
                if callable(attach_algorithmic_origin):
                    state = attach_algorithmic_origin(
                        self.material,
                        {
                            "plastic_strain": self.plastic_work[index],
                            "alpha": self.alpha_work[index],
                        },
                        state,
                        self.num_layers,
                        tangent_evaluated=bool(tangent),
                    )
                states[int(element_id)] = state
        else:
            points_per_element = self.n_gp * self.num_layers
            for index, element_id in enumerate(self.element_ids):
                start_layer = index * points_per_element
                stop_layer = start_layer + points_per_element
                existing = committed_states.get(int(element_id))
                if int(element_id) in deleted and isinstance(existing, dict):
                    states[int(element_id)] = existing
                else:
                    states[int(element_id)] = {
                        "plastic_strain": self.plastic_work[index].copy(),
                        "alpha": self.alpha_work[index].copy(),
                        "layer_strain": layer_strain[start_layer:stop_layer].copy(),
                    }

        if deleted:
            for index, element_id in enumerate(self.element_ids):
                if int(element_id) not in deleted:
                    continue
                F_batch[index] *= residual_fraction
                if tangent and K_batch is not None:
                    K_batch[index] *= residual_fraction

        return F_batch, K_batch if tangent else None, states, time.perf_counter() - start

    def persistent_state_eligibility(
        self,
        state_batch: ShellStateBatch,
    ) -> Tuple[bool, Optional[str]]:
        """Return whether this exact batch can consume persistent state arrays.

        Elastic Batch-B shells deliberately omit iteration state, and immutable
        initial fields still require the qualified scalar/exact override.  Both
        therefore remain on their existing explicit mapping paths.
        """

        if not self.has_plasticity:
            return False, "elastic_shell_has_no_mutable_constitutive_history"
        if not isinstance(state_batch, ShellStateBatch):
            return False, "state_input_is_legacy_mapping"
        return state_batch.persistent_eligibility(self.state_layout)

    def evaluate_persistent(
        self,
        displacements: np.ndarray,
        state_batch: ShellStateBatch,
        trial_token: StateTrialToken,
        tangent: bool,
        deleted_element_ids: Sequence[int] = (),
        residual_stiffness_fraction: float = 1.0,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        """Evaluate directly from/to solver-owned committed/trial arrays.

        This is a narrow production integration point for solver lifecycles that
        explicitly own a :class:`ShellStateBatch`.  It does not mutate or attach
        constitutive history to the cached assembly plan.  Existing callers keep
        using :meth:`evaluate`, which remains the public-dictionary fallback.
        """

        from .vectorized_nonlinear import batch_shell_nonlinear_response

        eligible, reason = self.persistent_state_eligibility(state_batch)
        if not eligible:
            raise PersistentStateEligibilityError(
                f"Persistent shell state is not eligible: {reason}"
            )
        state_batch.validate_trial_token(trial_token)
        start = time.perf_counter()
        batch_element_ids = set(self.state_layout.element_ids)
        deleted = tuple(
            int(element_id)
            for element_id in deleted_element_ids
            if int(element_id) in batch_element_ids
        )
        if deleted:
            state_batch.freeze_deleted(deleted)
        committed = state_batch.committed_arrays()
        self.u_work[:] = np.asarray(displacements, dtype=float)[self.dof_mappings]
        F_batch, K_batch, ep_new, alpha_new, layer_strain = (
            self._shell_kernel()(
                self.u_work,
                self.T0,
                self.B_m,
                self.B_b,
                self.B_d,
                self.Gw,
                self.detw,
                self.B_s,
                self.detw_shear,
                float(self.material.elastic_modulus),
                float(self.material.poisson_ratio),
                float(self.material.shear_modulus),
                self.thickness,
                self.drilling_stabilization,
                bool(tangent),
                getattr(self.material, "hardening_curve", None),
                committed.plastic_strain,
                committed.alpha,
                self.num_layers,
            )
        )
        if np.any(self.linear_corrections):
            F_batch += np.einsum(
                "eij,ej->ei",
                self.linear_corrections,
                self.u_work,
                optimize=True,
            )
            if tangent and K_batch is not None:
                K_batch += self.linear_corrections
        state_batch.update_trial(
            trial_token,
            plastic_strain=ep_new,
            alpha=alpha_new,
            layer_strain=layer_strain,
        )
        if tangent:
            q4_origin_indices = np.asarray(
                [
                    index
                    for index, element in enumerate(self.elements)
                    if str(
                        getattr(
                            element,
                            "current_state_algorithmic_origin_schema_id",
                            "",
                        )
                    ).startswith("E4_PL_Q4_ACCEPTED_DISCRETE_RETURN_MAP_ORIGIN_")
                ],
                dtype=np.intp,
            )
            if q4_origin_indices.size:
                state_batch.update_q4_algorithmic_origin_trial(
                    trial_token,
                    parent_plastic_strain=committed.plastic_strain[
                        q4_origin_indices
                    ],
                    parent_alpha=committed.alpha[q4_origin_indices],
                    element_ids=tuple(
                        int(self.element_ids[index])
                        for index in q4_origin_indices
                    ),
                )
        if deleted:
            deleted_set = set(deleted)
            residual_fraction = float(residual_stiffness_fraction)
            for index, element_id in enumerate(self.element_ids):
                if int(element_id) not in deleted_set:
                    continue
                F_batch[index] *= residual_fraction
                if tangent and K_batch is not None:
                    K_batch[index] *= residual_fraction
        return (
            F_batch,
            K_batch if tangent else None,
            time.perf_counter() - start,
        )


@dataclass
class _QuadraticBeamBatchPlan:
    """Persistent geometry and scatter data for elastic Beam3 elements."""

    element_ids: np.ndarray
    elements: Tuple[Any, ...]
    dof_mappings: np.ndarray
    force_positions: np.ndarray
    tangent_positions: np.ndarray
    transforms: np.ndarray
    strain_rows: np.ndarray
    jacobian_weights: np.ndarray
    properties: np.ndarray

    @classmethod
    def build(
        cls,
        model: "FEModel",
        items: Sequence[Tuple[int, Any, _ElementScatter]],
    ) -> "_QuadraticBeamBatchPlan":
        from .elements import _beam_material_properties

        count = len(items)
        element_ids = np.empty(count, dtype=np.int64)
        dof_mappings = np.empty((count, 18), dtype=np.intp)
        force_positions = np.empty((count, 18), dtype=np.intp)
        tangent_positions = np.empty((count, 18 * 18), dtype=np.intp)
        transforms = np.empty((count, 18, 18), dtype=float)
        strain_rows = np.zeros((count, 3, 8, 18), dtype=float)
        jacobian_weights = np.empty((count, 3), dtype=float)
        properties = np.empty((count, 6), dtype=float)
        elements: List[Any] = []
        for index, (element_id, element, scatter) in enumerate(items):
            coords = element.get_node_coordinates(model.mesh)
            length, transform = element._beam_frame_and_transform(coords)
            material = model.get_material(element.material_name)
            E, G12, G13, GJ = _beam_material_properties(material, element.cross_section)
            element_ids[index] = int(element_id)
            elements.append(element)
            dof_mappings[index] = scatter.dof_mapping
            force_positions[index] = scatter.force_positions
            tangent_positions[index] = scatter.tangent_positions
            transforms[index] = transform
            properties[index] = (
                E * element._A,
                E * element._Iy,
                E * element._Iz,
                G12 * element._A * element._ky,
                G13 * element._A * element._kz,
                GJ,
            )
            for gp_index, (xi, weight) in enumerate(
                zip(element.GAUSS_POINTS, element.GAUSS_WEIGHTS)
            ):
                shape, derivative_xi = element.compute_shape_functions(float(xi))
                derivative_x = derivative_xi * 2.0 / float(length)
                for node_index in range(3):
                    base = node_index * 6
                    strain_rows[index, gp_index, 0, base] = derivative_x[node_index]
                    strain_rows[index, gp_index, 1, base + 1] = derivative_x[node_index]
                    strain_rows[index, gp_index, 2, base + 2] = derivative_x[node_index]
                    strain_rows[index, gp_index, 3, base + 4] = derivative_x[node_index]
                    strain_rows[index, gp_index, 4, base + 5] = derivative_x[node_index]
                    strain_rows[index, gp_index, 5, base + 3] = derivative_x[node_index]
                    strain_rows[index, gp_index, 6, base + 1] = derivative_x[node_index]
                    strain_rows[index, gp_index, 6, base + 5] = -shape[node_index]
                    strain_rows[index, gp_index, 7, base + 2] = derivative_x[node_index]
                    strain_rows[index, gp_index, 7, base + 4] = shape[node_index]
                jacobian_weights[index, gp_index] = float(length) / 2.0 * float(weight)
        return cls(
            element_ids=element_ids,
            elements=tuple(elements),
            dof_mappings=dof_mappings,
            force_positions=force_positions,
            tangent_positions=tangent_positions,
            transforms=transforms,
            strain_rows=strain_rows,
            jacobian_weights=jacobian_weights,
            properties=properties,
        )

    def evaluate_into_buffers(
        self,
        plan: "NonlinearAssemblyPlan",
        displacements: np.ndarray,
        committed_states: Mapping[int, Any],
        tangent: bool,
        trial_states: Dict[int, Any],
        deleted_element_ids: Sequence[int] = (),
        residual_stiffness_fraction: float = 1.0,
    ) -> float:
        from .vectorized_beam import quadratic_beam_nonlinear_into_buffers_jit

        start = time.perf_counter()
        quadratic_beam_nonlinear_into_buffers_jit(
            np.asarray(displacements, dtype=float),
            self.dof_mappings,
            self.transforms,
            self.strain_rows,
            self.jacobian_weights,
            self.properties,
            self.force_positions,
            self.tangent_positions,
            plan.force_values,
            plan.tangent_values,
            bool(tangent),
        )
        for element_id in self.element_ids:
            state = committed_states.get(int(element_id))
            if state is not None:
                trial_states[int(element_id)] = state
        deleted = {int(element_id) for element_id in deleted_element_ids}
        if deleted:
            model = plan.model
            for index, (element_id, element) in enumerate(zip(self.element_ids, self.elements)):
                element_key = int(element_id)
                if element_key not in deleted:
                    continue
                state = committed_states.get(element_key)
                require_legacy_direct_nonlinear_element(
                    element,
                    context="quadratic-beam deleted-element fallback",
                )
                force, stiffness, trial_state = element.compute_nonlinear_response(
                    model.mesh,
                    model.get_material(element.material_name),
                    np.asarray(displacements, dtype=float)[self.dof_mappings[index]],
                    state,
                    plan.num_layers,
                    tangent,
                )
                plan.force_values[self.force_positions[index]] = (
                    float(residual_stiffness_fraction)
                    * np.asarray(force, dtype=float).reshape(-1)
                )
                if tangent and stiffness is not None:
                    plan.tangent_values[self.tangent_positions[index]] = (
                        float(residual_stiffness_fraction)
                        * np.asarray(stiffness, dtype=float).reshape(-1)
                    )
                if state is not None:
                    trial_states[element_key] = state
                elif trial_state is not None:
                    trial_states[element_key] = trial_state
        return time.perf_counter() - start


@dataclass
class NonlinearAssemblyPlan:
    """Persistent nonlinear global assembly data for one FE model."""

    model_ref: "weakref.ReferenceType[FEModel]"
    num_layers: int
    revision: Tuple[int, ...]
    qualified_input_signature: Tuple[Any, ...]
    shell_batches: Tuple[_ShellBatchPlan, ...]
    quadratic_beam_batches: Tuple[_QuadraticBeamBatchPlan, ...]
    non_shell_elements: Tuple[_ElementScatter, ...]
    force_dofs_flat: np.ndarray
    tangent_scatter: np.ndarray
    csr_indptr: np.ndarray
    csr_indices: np.ndarray
    force_values: np.ndarray
    tangent_values: np.ndarray
    setup_seconds: float
    timings: NonlinearAssemblyTimings = field(default_factory=NonlinearAssemblyTimings)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @property
    def model(self) -> "FEModel":
        model = self.model_ref()
        if model is None:
            raise RuntimeError("The FE model associated with this assembly plan no longer exists")
        return model

    @property
    def total_dofs(self) -> int:
        return int(self.csr_indptr.size - 1)

    @property
    def nnz(self) -> int:
        return int(self.csr_indices.size)

    @classmethod
    def build(cls, model: "FEModel", num_layers: int) -> "NonlinearAssemblyPlan":
        from .elements import QuadraticBeamElement, ShellElement
        from .materials import is_isotropic_material, is_orthotropic_material
        from .vectorized_nonlinear import shell_nonlinear_batch_eligible

        start = time.perf_counter()
        mesh = model.mesh
        total_dofs = int(mesh.dof_manager.total_dofs)
        element_records: List[Tuple[int, Any, np.ndarray, int, int, int, int]] = []
        all_rows: List[np.ndarray] = []
        all_cols: List[np.ndarray] = []
        force_offset = 0
        tangent_offset = 0

        for element_id, element in mesh.elements.items():
            mapping = np.asarray(element.get_dof_mapping(mesh), dtype=np.intp).reshape(-1)
            if mapping.size == 0:
                continue
            n_local = int(mapping.size)
            rows = np.repeat(mapping, n_local)
            cols = np.tile(mapping, n_local)
            all_rows.append(rows)
            all_cols.append(cols)
            element_records.append(
                (
                    int(element_id),
                    element,
                    mapping,
                    force_offset,
                    force_offset + n_local,
                    tangent_offset,
                    tangent_offset + n_local * n_local,
                )
            )
            force_offset += n_local
            tangent_offset += n_local * n_local

        rows_concat = np.concatenate(all_rows) if all_rows else np.empty(0, dtype=np.intp)
        cols_concat = np.concatenate(all_cols) if all_cols else np.empty(0, dtype=np.intp)
        if rows_concat.size:
            pair_keys = rows_concat.astype(np.int64) * np.int64(total_dofs) + cols_concat.astype(np.int64)
            unique_keys, tangent_scatter = np.unique(pair_keys, return_inverse=True)
            unique_rows = (unique_keys // np.int64(total_dofs)).astype(np.intp)
            unique_cols = (unique_keys % np.int64(total_dofs)).astype(np.intp)
            row_counts = np.bincount(unique_rows, minlength=total_dofs)
            csr_indptr = np.empty(total_dofs + 1, dtype=np.intp)
            csr_indptr[0] = 0
            np.cumsum(row_counts, out=csr_indptr[1:])
            csr_indices = unique_cols
        else:
            tangent_scatter = np.empty(0, dtype=np.intp)
            csr_indptr = np.zeros(total_dofs + 1, dtype=np.intp)
            csr_indices = np.empty(0, dtype=np.intp)

        scatter_records: Dict[int, _ElementScatter] = {}
        force_dofs: List[np.ndarray] = []
        for element_id, element, mapping, f0, f1, k0, k1 in element_records:
            force_positions = np.arange(f0, f1, dtype=np.intp)
            tangent_positions = np.arange(k0, k1, dtype=np.intp)
            scatter = _ElementScatter(
                element_id=element_id,
                element=element,
                dof_mapping=mapping,
                force_positions=force_positions,
                tangent_positions=tangent_positions,
            )
            scatter_records[element_id] = scatter
            force_dofs.append(mapping)

        shell_groups: Dict[Tuple[Any, ...], List[Tuple[int, Any, _ElementScatter]]] = {}
        quadratic_beams: List[Tuple[int, Any, _ElementScatter]] = []
        non_shell: List[_ElementScatter] = []
        for element_id, element, *_rest in element_records:
            scatter = scatter_records[element_id]
            material = model.get_material(element.material_name)
            ordinary_shell_batch = (
                isinstance(element, ShellElement)
                and shell_nonlinear_batch_eligible(element)
                and is_isotropic_material(material)
            )
            extended_elastic_s4_batch = bool(
                _EXTENDED_ELASTIC_S4_BATCHES
                and isinstance(element, ShellElement)
                and bool(getattr(element, "_is_4node", False))
                and not (
                    bool(getattr(element, "_is_8node", False))
                    and bool(getattr(element, "reduced_integration", False))
                )
                and (
                    getattr(element, "shell_section", None) is not None
                    or is_orthotropic_material(material)
                )
                and getattr(material, "hardening_curve", None) is None
                and getattr(material, "hill_yield", None) is None
            )
            if ordinary_shell_batch or extended_elastic_s4_batch:
                shell_kind = (
                    "generalized_s4"
                    if getattr(element, "shell_section", None) is not None
                    else (
                        "orthotropic_s4"
                        if is_orthotropic_material(material)
                        else "isotropic"
                    )
                )
                if shell_kind == "isotropic" and _q4_assumed_strain.route(
                    element, mesh, material
                ):
                    shell_kind = _Q4_ASSUMED_STRAIN_SHELL_KIND
                section = getattr(element, "shell_section", None)
                section_key = (
                    None
                    if section is None
                    else (
                        np.asarray(section.A, dtype=float).tobytes(),
                        np.asarray(section.B, dtype=float).tobytes(),
                        np.asarray(section.D, dtype=float).tobytes(),
                        np.asarray(section.As, dtype=float).tobytes(),
                    )
                )
                key = (
                    shell_kind,
                    int(element.num_nodes),
                    float(element.thickness),
                    float(element.drilling_stabilization),
                    str(element.material_name),
                    section_key,
                )
                shell_groups.setdefault(key, []).append((element_id, element, scatter))
            elif (
                isinstance(element, QuadraticBeamElement)
                and element.generalized_section is None
                and element._fiber_plasticity_config(material) is None
            ):
                quadratic_beams.append((element_id, element, scatter))
            else:
                non_shell.append(scatter)

        shell_batches = tuple(
            _ShellBatchPlan.build(model, key, items, int(num_layers))
            for key, items in shell_groups.items()
        )
        quadratic_beam_batches = (
            (_QuadraticBeamBatchPlan.build(model, quadratic_beams),)
            if quadratic_beams
            else ()
        )

        return cls(
            model_ref=weakref.ref(model),
            num_layers=int(num_layers),
            revision=_revision_tuple(mesh),
            qualified_input_signature=_qualified_plan_input_signature(model),
            shell_batches=shell_batches,
            quadratic_beam_batches=quadratic_beam_batches,
            non_shell_elements=tuple(non_shell),
            force_dofs_flat=np.concatenate(force_dofs) if force_dofs else np.empty(0, dtype=np.intp),
            tangent_scatter=np.asarray(tangent_scatter, dtype=np.intp),
            csr_indptr=csr_indptr,
            csr_indices=csr_indices,
            force_values=np.zeros(force_offset, dtype=float),
            tangent_values=np.zeros(tangent_offset, dtype=float),
            setup_seconds=float(time.perf_counter() - start),
        )

    def is_valid(self, model: "FEModel", num_layers: int) -> bool:
        return (
            self.model_ref() is model
            and int(num_layers) == self.num_layers
            and _revision_tuple(model.mesh) == self.revision
            and _qualified_plan_input_signature(model)
            == self.qualified_input_signature
        )

    def assemble(
        self,
        displacements: np.ndarray,
        committed_states: Mapping[int, Any],
        tangent: bool = True,
        deleted_element_ids: Sequence[int] = (),
        residual_stiffness_fraction: float = 1.0,
    ) -> Tuple[np.ndarray, Optional[sparse.csr_matrix], Dict[int, Any]]:
        """Assemble internal force and tangent using persistent batch/scatter data."""
        with self._lock:
            state_token = begin_state_evaluation(
                committed_states,
                model=self.model,
                displacements=displacements,
            )
            start_total = time.perf_counter()
            self.timings.calls += 1
            if tangent:
                self.timings.tangent_calls += 1
            else:
                self.timings.residual_only_calls += 1

            self.force_values.fill(0.0)
            if tangent:
                self.tangent_values.fill(0.0)
            trial_states: Dict[int, Any] = {}
            deleted = {int(element_id) for element_id in deleted_element_ids}
            residual_fraction = float(residual_stiffness_fraction)
            freeze_deleted = getattr(committed_states, "freeze_deleted", None)
            if deleted and callable(freeze_deleted):
                freeze_deleted(tuple(sorted(deleted)))

            state_start = time.perf_counter()
            for batch in self.shell_batches:
                used_persistent_state = False
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
                    F_batch, K_batch, kernel_seconds = batch.evaluate_persistent(
                        displacements,
                        persistent_trial[0],
                        persistent_trial[1],
                        tangent,
                        deleted_element_ids=tuple(deleted),
                        residual_stiffness_fraction=residual_fraction,
                    )
                    batch_states = {}
                else:
                    F_batch, K_batch, batch_states, kernel_seconds = batch.evaluate(
                        displacements,
                        committed_states,
                        tangent,
                        deleted_element_ids=tuple(deleted),
                        residual_stiffness_fraction=residual_fraction,
                    )
                self.timings.shell_kernel_seconds += kernel_seconds
                self.force_values[batch.force_positions.reshape(-1)] = np.asarray(F_batch, dtype=float).reshape(-1)
                if tangent and K_batch is not None:
                    self.tangent_values[batch.tangent_positions.reshape(-1)] = np.asarray(K_batch, dtype=float).reshape(-1)
                trial_states.update(batch_states)
                if not used_persistent_state:
                    override_start = time.perf_counter()
                    override_count = _apply_initial_field_shell_overrides(
                        self,
                        batch,
                        displacements,
                        committed_states,
                        tangent,
                        trial_states,
                        deleted_element_ids=tuple(deleted),
                        residual_stiffness_fraction=residual_fraction,
                    )
                    self.timings.initial_field_override_seconds += time.perf_counter() - override_start
                    self.timings.initial_field_override_elements += override_count
            self.timings.state_pack_seconds += time.perf_counter() - state_start

            for batch in self.quadratic_beam_batches:
                self.timings.non_shell_seconds += batch.evaluate_into_buffers(
                    self,
                    displacements,
                    committed_states,
                    tangent,
                    trial_states,
                    deleted_element_ids=tuple(deleted),
                    residual_stiffness_fraction=residual_fraction,
                )

            non_shell_start = time.perf_counter()
            model = self.model
            mesh = model.mesh
            for record in self.non_shell_elements:
                material = model.get_material(record.element.material_name)
                u_element = np.asarray(displacements, dtype=float)[record.dof_mapping]
                f_element, k_element, trial_state = evaluate_nonlinear_element(
                    record.element,
                    mesh,
                    material,
                    u_element,
                    committed_states.get(record.element_id),
                    self.num_layers,
                    tangent,
                    committed_states=committed_states,
                    state_token=state_token,
                    element_id=record.element_id,
                    kinematics="von_karman",
                )
                if record.element_id in deleted:
                    f_element = residual_fraction * np.asarray(f_element, dtype=float)
                    if tangent and k_element is not None:
                        k_element = residual_fraction * np.asarray(k_element, dtype=float)
                    if record.element_id in committed_states:
                        trial_state = committed_states[record.element_id]
                self.force_values[record.force_positions] = np.asarray(f_element, dtype=float).reshape(-1)
                if tangent and k_element is not None:
                    self.tangent_values[record.tangent_positions] = np.asarray(k_element, dtype=float).reshape(-1)
                if trial_state is not None:
                    trial_states[record.element_id] = trial_state
            self.timings.non_shell_seconds += time.perf_counter() - non_shell_start

            scatter_start = time.perf_counter()
            force = _scatter_sum(self.force_values, self.force_dofs_flat, self.total_dofs)
            self.timings.force_scatter_seconds += time.perf_counter() - scatter_start

            tangent_matrix: Optional[sparse.csr_matrix]
            if tangent:
                scatter_start = time.perf_counter()
                csr_data = _scatter_sum(self.tangent_values, self.tangent_scatter, self.nnz)
                tangent_matrix = sparse.csr_matrix(
                    (csr_data, self.csr_indices, self.csr_indptr),
                    shape=(self.total_dofs, self.total_dofs),
                    copy=False,
                )
                self.timings.tangent_scatter_seconds += time.perf_counter() - scatter_start
            else:
                tangent_matrix = None

            self.timings.total_seconds += time.perf_counter() - start_total
            state_payload = finish_state_evaluation(
                committed_states,
                state_token,
                trial_states,
            )
            return force, tangent_matrix, state_payload

    def diagnostics(self) -> Dict[str, Any]:
        from .elements import ShellElement
        from .materials import is_orthotropic_material

        fallback_reasons: Dict[str, List[int]] = {}
        for record in self.non_shell_elements:
            if not isinstance(record.element, ShellElement):
                continue
            if getattr(record.element, "shell_section", None) is not None:
                reason = "generalized_shell_section"
            elif is_orthotropic_material(
                self.model.get_material(record.element.material_name)
            ):
                reason = "orthotropic_material"
            else:
                continue
            fallback_reasons.setdefault(reason, []).append(int(record.element_id))
        for ids in fallback_reasons.values():
            ids.sort()
        fallback_ids = sorted(
            element_id
            for ids in fallback_reasons.values()
            for element_id in ids
        )
        fallback_reason = next(iter(fallback_reasons), None)
        if len(fallback_reasons) > 1:
            fallback_reason = "mixed_constitutive"
        return {
            "num_layers": int(self.num_layers),
            "revision": list(self.revision),
            "shell_batch_count": len(self.shell_batches),
            "shell_element_count": int(sum(batch.element_ids.size for batch in self.shell_batches)),
            "quadratic_beam_batch_count": len(self.quadratic_beam_batches),
            "quadratic_beam_element_count": int(
                sum(batch.element_ids.size for batch in self.quadratic_beam_batches)
            ),
            "non_shell_element_count": len(self.non_shell_elements),
            "constitutive_fallback": (
                None
                if not fallback_ids
                else {
                    "path": "general_element",
                    "reason": fallback_reason,
                    "element_ids": fallback_ids,
                    **(
                        {"reasons": fallback_reasons}
                        if len(fallback_reasons) > 1
                        else {}
                    ),
                }
            ),
            "total_dofs": self.total_dofs,
            "tangent_nnz": self.nnz,
            "local_force_entries": int(self.force_values.size),
            "local_tangent_entries": int(self.tangent_values.size),
            "setup_seconds": float(self.setup_seconds),
            "timings": self.timings.to_dict(),
        }


_PLAN_CACHE: "weakref.WeakKeyDictionary[FEModel, Dict[int, NonlinearAssemblyPlan]]" = weakref.WeakKeyDictionary()
_CACHE_LOCK = threading.RLock()
_ORIGINAL_ASSEMBLER = None
_ORIGINAL_SPARSITY_GETTER = None
_ORIGINAL_DISPLACEMENT_SOLVER = None
_INSTALLED = False


def get_nonlinear_assembly_plan(model: "FEModel", num_layers: int) -> NonlinearAssemblyPlan:
    """Return a valid cached plan or build a new one."""
    with _CACHE_LOCK:
        by_layers = _PLAN_CACHE.setdefault(model, {})
        plan = by_layers.get(int(num_layers))
        if plan is None or not plan.is_valid(model, int(num_layers)):
            plan = NonlinearAssemblyPlan.build(model, int(num_layers))
            by_layers[int(num_layers)] = plan
        return plan


def clear_nonlinear_assembly_cache(model: Optional["FEModel"] = None) -> None:
    with _CACHE_LOCK:
        if model is None:
            _PLAN_CACHE.clear()
        else:
            _PLAN_CACHE.pop(model, None)


def nonlinear_assembly_diagnostics(model: Optional["FEModel"] = None) -> Dict[str, Any]:
    with _CACHE_LOCK:
        if model is not None:
            plans = _PLAN_CACHE.get(model, {})
            return {str(layers): plan.diagnostics() for layers, plan in plans.items()}
        result: Dict[str, Any] = {}
        for cached_model, plans in list(_PLAN_CACHE.items()):
            result[str(id(cached_model))] = {
                "model_name": getattr(cached_model, "name", None),
                "plans": {str(layers): plan.diagnostics() for layers, plan in plans.items()},
            }
        return result


def _requires_deleted_qualified_s3_reference_assembly(
    model: "FEModel",
    committed_states: Mapping[int, Any],
    deleted_element_ids: Sequence[int],
) -> bool:
    """Whether frozen native S3 history needs its isolated residual route."""

    from .e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement

    for raw_element_id in deleted_element_ids:
        element_id = int(raw_element_id)
        element = model.mesh.get_element(element_id)
        if type(element) is NativeParityE4PLS3V2DShellElement:
            # Exact V2D deletion uses its validated frozen-history operator;
            # the generic plan only scales an ordinary active-state trial.
            return True
        if (
            element is None
            or str(getattr(element, "formulation_id", ""))
            != "E4_PL_QUALIFIED_S3_COMPANION_V1"
            or not callable(
                getattr(
                    element,
                    "compute_noncurrent_deleted_residual_operator",
                    None,
                )
            )
        ):
            continue
        state = committed_states.get(element_id)
        digest = (
            state.get("state_integrity_sha256")
            if isinstance(state, Mapping)
            else None
        )
        if (
            isinstance(state, Mapping)
            and state.get("formulation_id")
            == "E4_PL_QUALIFIED_S3_COMPANION_V1"
            and state.get("element_id") == element_id
            and isinstance(digest, str)
            and len(digest) == 64
        ):
            return True
    return False


def _optimized_assemble_nonlinear_system(
    model: "FEModel",
    displacements: np.ndarray,
    committed_states: Dict[int, Any],
    num_layers: int,
    tangent: bool = True,
    deleted_element_ids: Optional[Sequence[int]] = None,
    residual_stiffness_fraction: float = 1.0,
    **extra,
):
    kinematics = str(extra.pop("kinematics", "von_karman"))
    corotational_tangent = str(
        extra.pop("corotational_tangent", "rotated")
    )
    deleted = tuple(int(value) for value in (deleted_element_ids or ()))
    deleted_s3_reference = _requires_deleted_qualified_s3_reference_assembly(
        model,
        committed_states,
        deleted,
    )
    from .general_contact.runtime import active_contact_manager

    if (active_contact_manager() is not None or extra
            or kinematics != "von_karman" or deleted_s3_reference):
        # The assembly plan encodes the von Karman element response; other
        # kinematics and per-element scale options use the scalar reference
        # assembler. Immutable initial fields are handled by exact per-element
        # overrides inside the otherwise accelerated shell batches.
        assembler = _ORIGINAL_ASSEMBLER
        if assembler is None:
            from . import nonlinear_static as _nonlinear_static

            assembler = _nonlinear_static._assemble_nonlinear_system
        return assembler(
            model,
            displacements,
            committed_states,
            num_layers,
            tangent=tangent,
            deleted_element_ids=deleted,
            residual_stiffness_fraction=float(residual_stiffness_fraction),
            kinematics=kinematics,
            corotational_tangent=corotational_tangent,
            **extra,
        )
    assembly_start = time.perf_counter()
    plan = get_nonlinear_assembly_plan(model, int(num_layers))
    try:
        result = plan.assemble(
            displacements,
            committed_states,
            tangent=tangent,
            deleted_element_ids=deleted,
            residual_stiffness_fraction=float(residual_stiffness_fraction),
        )
        record_nonlinear_assembly_execution(
            path="persistent_full_coordinate",
            tangent=bool(tangent),
            elapsed_seconds=time.perf_counter() - assembly_start,
            plan=plan,
        )
        return result
    except BaseException:
        # Batch A/B plans open the same solver-owned candidate transaction as
        # the reference assembler.  Keep retries and cancellation reusable
        # when an element, compiled kernel, or post-assembly observer fails.
        discard_active_state_candidate(committed_states)
        raise


def _revision_cached_sparsity_pattern(mesh: "FEMesh", matrix_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """Cached COO row/column pattern using mesh revisions rather than JSON/SHA."""
    if not hasattr(mesh, "_sparsity_cache"):
        mesh._sparsity_cache = {}
    signature = _sparsity_revision_tuple(mesh)
    cached = mesh._sparsity_cache.get(matrix_type)
    if cached is not None and cached.get("signature") == signature:
        return cached["rows"], cached["cols"]

    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    for element in mesh.elements.values():
        mapping = np.asarray(element.get_dof_mapping(mesh), dtype=np.intp).reshape(-1)
        if mapping.size == 0:
            continue
        rows.append(np.repeat(mapping, mapping.size))
        cols.append(np.tile(mapping, mapping.size))
    rows_concat = np.concatenate(rows) if rows else np.empty(0, dtype=np.intp)
    cols_concat = np.concatenate(cols) if cols else np.empty(0, dtype=np.intp)
    mesh._sparsity_cache[matrix_type] = {
        "rows": rows_concat,
        "cols": cols_concat,
        "signature": signature,
    }
    return rows_concat, cols_concat


def _solve_static_displacement_control_block(
    *,
    model,
    T,
    u0,
    F_const,
    F_prop,
    stage_vectors,
    load_case,
    constant_load_case,
    load_program,
    displacement_control,
    committed_states,
    num_layers,
    num_steps,
    max_iterations,
    tolerance,
    info,
    start_time,
    resource_config=None,
    kinematics="von_karman",
    corotational_tangent="not_applicable",
    initial_reduced_displacements=None,
    cancellation_token=None,
    progress_callback=None,
    record_increment_snapshots=False,
    initial_load_factor=0.0,
    recover_initial_load_factor=False,
    step_index_offset=0,
    initial_steps=(),
    initial_history=(),
    initial_total_iterations=0,
    restart_analysis_contract=None,
    rotating_mpc_map=None,
    _exact_guard=None,
):
    """Displacement control using block elimination on the structural tangent."""
    from . import nonlinear_static as ns
    from .cases import make_result_case
    from .jit_compiler import numba_thread_scope
    from .linalg import MatrixClass, factorize

    exact_guard = _exact_guard or ns._EXACT_QUALIFIED_LIFECYCLE_GUARD
    exact_guard(
        model,
        context="nonlinear displacement-control block preflight",
    )

    from .general_contact.runtime import active_contact_manager

    checkpoint_path = (
        active_contact_manager() is not None
        or restart_analysis_contract is not None
        or rotating_mpc_map is not None
        or float(initial_load_factor) != 0.0
        or bool(recover_initial_load_factor)
        or int(step_index_offset) != 0
        or bool(initial_steps)
        or bool(initial_history)
        or int(initial_total_iterations) != 0
    )
    if checkpoint_path:
        # Contact and checkpoint continuation use one transaction-aware path.
        # The block solver does not own contact gap, sweep, or tool history.
        solver = _ORIGINAL_DISPLACEMENT_SOLVER
        if solver is None:
            raise RuntimeError(
                "reference displacement-control solver is unavailable for restart"
            )
        return solver(
            model=model,
            T=T,
            u0=u0,
            F_const=F_const,
            F_prop=F_prop,
            stage_vectors=stage_vectors,
            load_case=load_case,
            constant_load_case=constant_load_case,
            load_program=load_program,
            displacement_control=displacement_control,
            committed_states=committed_states,
            num_layers=num_layers,
            num_steps=num_steps,
            max_iterations=max_iterations,
            tolerance=tolerance,
            info=info,
            start_time=start_time,
            resource_config=resource_config,
            kinematics=kinematics,
            corotational_tangent=corotational_tangent,
            initial_reduced_displacements=initial_reduced_displacements,
            cancellation_token=cancellation_token,
            progress_callback=progress_callback,
            record_increment_snapshots=record_increment_snapshots,
            initial_load_factor=initial_load_factor,
            recover_initial_load_factor=recover_initial_load_factor,
            step_index_offset=step_index_offset,
            initial_steps=initial_steps,
            initial_history=initial_history,
            initial_total_iterations=initial_total_iterations,
            restart_analysis_contract=restart_analysis_contract,
            rotating_mpc_map=rotating_mpc_map,
            _exact_guard=exact_guard,
        )

    follower_active = ns._has_follower_pressure(
        load_case,
        model=model,
        _exact_guard=exact_guard,
    ) or ns._has_follower_pressure(
        constant_load_case,
        model=model,
        _exact_guard=exact_guard,
    )
    if load_program is not None:
        follower_active = follower_active or any(
            ns._has_follower_pressure(
                stage.load_case,
                model=model,
                _exact_guard=exact_guard,
            )
            for stage in load_program.stages
        )
    exact_guard(
        model,
        context="nonlinear displacement-control block load observation",
    )
    if str(kinematics) != "von_karman" or follower_active:
        # The block-elimination fast path encodes the von Karman assembly plan;
        # corotational displacement control uses the original solver.
        solver = _ORIGINAL_DISPLACEMENT_SOLVER
        if solver is None:
            solver = ns._solve_static_displacement_control
        return solver(
            model=model,
            T=T,
            u0=u0,
            F_const=F_const,
            F_prop=F_prop,
            stage_vectors=stage_vectors,
            load_case=load_case,
            constant_load_case=constant_load_case,
            load_program=load_program,
            displacement_control=displacement_control,
            committed_states=committed_states,
            num_layers=num_layers,
            num_steps=num_steps,
            max_iterations=max_iterations,
            tolerance=tolerance,
            info=info,
            start_time=start_time,
            resource_config=resource_config,
            kinematics=kinematics,
            corotational_tangent=corotational_tangent,
            initial_reduced_displacements=initial_reduced_displacements,
            cancellation_token=cancellation_token,
            progress_callback=progress_callback,
            record_increment_snapshots=record_increment_snapshots,
            initial_load_factor=initial_load_factor,
            recover_initial_load_factor=recover_initial_load_factor,
            step_index_offset=step_index_offset,
            initial_steps=initial_steps,
            initial_history=initial_history,
            initial_total_iterations=initial_total_iterations,
            restart_analysis_contract=restart_analysis_contract,
            rotating_mpc_map=rotating_mpc_map,
            _exact_guard=exact_guard,
        )

    if load_program is not None:
        if len(load_program.stages) == 1:
            F_const_dc = F_const
            F_prop_dc = load_program.stages[0].target_factor * stage_vectors[0]
            active_stage = load_program.stages[0].name
        else:
            F_const_dc = F_const.copy()
            for stage, vector in zip(load_program.stages[:-1], stage_vectors[:-1]):
                F_const_dc += stage.target_factor * vector
            F_prop_dc = load_program.stages[-1].target_factor * stage_vectors[-1]
            active_stage = load_program.stages[-1].name
        info["displacement_control_load_split"] = {
            "constant_stages": [stage.name for stage in load_program.stages[:-1]],
            "proportional_stage": active_stage,
        }
    else:
        F_const_dc = F_const
        F_prop_dc = F_prop
        active_stage = "displacement_control"

    n_red = int(T.shape[1])
    if initial_reduced_displacements is None:
        q = np.zeros(n_red, dtype=float)
    else:
        q = np.asarray(initial_reduced_displacements, dtype=float).reshape(-1).copy()
        if q.size != n_red:
            raise ValueError(
                f"initial_reduced_displacements has {q.size} entries; expected {n_red}"
            )
    lam = 0.0
    steps = []
    snapshots = []
    history: List[Dict[str, Any]] = []
    status = "completed"
    failure_reason: Optional[str] = None
    total_iterations = 0
    reaction_force_reuse_count = 0
    reaction_force_reassembly_count = 0

    row_full = displacement_control.full_row(model)
    exact_guard(
        model,
        context="nonlinear displacement-control block row observation",
    )
    row_red = np.asarray(row_full @ T, dtype=float).reshape(-1)
    row_u0 = float(row_full @ u0)
    if float(np.linalg.norm(row_red)) <= 0.0:
        raise ValueError("Displacement control target is fixed or dependent and cannot be used as an unknown")

    F_const_red = np.asarray(T.T @ F_const_dc, dtype=float).reshape(-1)
    F_prop_red = np.asarray(T.T @ F_prop_dc, dtype=float).reshape(-1)
    if float(np.linalg.norm(F_prop_red)) <= 0.0:
        raise ValueError("Displacement control requires a non-zero proportional load vector")

    target_total = float(displacement_control.target_displacement)
    initial_control = float(row_red @ q + row_u0)
    target_increment = target_total - initial_control
    target_scale = max(abs(target_increment), abs(target_total), 1.0e-9)
    info["displacement_control_initial_value"] = initial_control

    assembly_threads = None if resource_config is None else resource_config.assembly_threads
    with numba_thread_scope(assembly_threads):
        for step_index in range(1, num_steps + 1):
            ns.cancellation_safe_point(
                cancellation_token,
                f"nonlinear_static.displacement.step:{step_index}",
            )
            exact_guard(
                model,
                context="nonlinear displacement-control block step cancellation",
            )
            q_step_start = q.copy()
            lam_step_start = float(lam)
            target = initial_control + target_increment * step_index / num_steps
            residual_norm = float("inf")
            constraint_error = float("inf")
            states_new = committed_states

            for iteration in range(1, max_iterations + 1):
                ns.cancellation_safe_point(
                    cancellation_token,
                    f"nonlinear_static.displacement.step:{step_index}.iteration:{iteration}",
                )
                exact_guard(
                    model,
                    context=(
                        "nonlinear displacement-control block iteration "
                        "cancellation"
                    ),
                )
                total_iterations += 1
                u = np.asarray(T @ q + u0, dtype=float).reshape(-1)
                F_int, K_T, trial_states = ns._assemble_nonlinear_system(
                    model, u, committed_states, num_layers
                )
                exact_guard(
                    model,
                    context="nonlinear displacement-control block assembly",
                )
                residual = F_const_red + lam * F_prop_red - np.asarray(T.T @ F_int, dtype=float).reshape(-1)
                residual_norm = float(np.linalg.norm(residual))
                current = float(row_red @ q + row_u0)
                constraint = target - current
                constraint_error = abs(constraint)
                reference = max(
                    float(np.linalg.norm(F_const_red + max(abs(lam), 1.0) * F_prop_red)),
                    1.0,
                )

                if residual_norm <= tolerance * reference and constraint_error <= tolerance * target_scale:
                    states_new = trial_states
                    break

                K_red = (T.T @ K_T @ T).tocsr()
                try:
                    handle = factorize(
                        K_red,
                        MatrixClass.SYMMETRIC_INDEFINITE,
                        signature=f"nonlinear.displacement_control.block:{step_index}:{iteration}",
                    )
                    rhs = np.column_stack((residual, F_prop_red))
                    solutions = np.asarray(handle.solve_many(rhs), dtype=float)
                    fixed_load_correction = solutions[:, 0]
                    load_direction = solutions[:, 1]
                except Exception:
                    failure_reason = "singular_structural_tangent"
                    break

                denominator = float(row_red @ load_direction)
                denominator_scale = max(float(np.linalg.norm(row_red)) * float(np.linalg.norm(load_direction)), 1.0)
                if abs(denominator) <= 1.0e-14 * denominator_scale:
                    failure_reason = "singular_displacement_constraint"
                    break
                d_lambda = (constraint - float(row_red @ fixed_load_correction)) / denominator
                d_q = fixed_load_correction + load_direction * d_lambda
                if np.any(~np.isfinite(d_q)) or not np.isfinite(d_lambda):
                    failure_reason = "nonfinite_block_solution"
                    break
                q += d_q
                lam += float(d_lambda)
            else:
                failure_reason = "maximum_iterations_reached"

            if failure_reason is not None:
                ns._discard_nonlinear_state_candidate(committed_states)
                q = q_step_start
                lam = lam_step_start
                status = "stopped_at_limit" if steps else "diverged"
                break

            committed_states = ns._commit_nonlinear_state_candidate(
                committed_states,
                states_new,
                model=model,
                accepted_displacements=np.asarray(T @ q + u0, dtype=float).reshape(-1),
            )
            exact_guard(
                model,
                context="nonlinear displacement-control block commit",
            )
            u = np.asarray(T @ q + u0, dtype=float).reshape(-1)
            current = float(row_red @ q + row_u0)
            # The converged Newton evaluation already contains the exact
            # accepted internal force.  Direct reduced assembly carries a
            # generation-qualified view of its element-local force buffers,
            # so expand that payload while it is still current instead of
            # repeating every constitutive evaluation.  If the payload is
            # stale or otherwise unrecognized, conservatively recover the
            # force through the original full-coordinate path.
            from .nonlinear_performance_batch_c import (
                materialize_full_internal_force,
            )

            reaction_internal = materialize_full_internal_force(
                F_int,
                model.mesh.dof_manager.total_dofs,
            )
            exact_guard(
                model,
                context="nonlinear displacement-control block reaction force",
            )
            if reaction_internal is None:
                reaction_internal, _unused, _reaction_states = (
                    ns._assemble_nonlinear_system(
                        model,
                        u,
                        committed_states,
                        num_layers,
                        tangent=False,
                        kinematics=kinematics,
                        corotational_tangent=corotational_tangent,
                        require_full_coordinates=True,
                    )
                )
                exact_guard(
                    model,
                    context=(
                        "nonlinear displacement-control block reaction "
                        "assembly"
                    ),
                )
                # Reaction recovery is diagnostic-only; do not leave its
                # trial constitutive state active for the next increment.
                ns._discard_nonlinear_state_candidate(committed_states)
                reaction_force_reassembly_count += 1
            else:
                reaction_force_reuse_count += 1
            support_reactions = ns._support_reaction_resultants(
                model,
                reaction_internal - (F_const_dc + lam * F_prop_dc),
            )
            exact_guard(
                model,
                context="nonlinear displacement-control block reactions",
            )
            steps.append(
                ns.NonlinearStaticStep(
                    step_index=step_index,
                    load_factor=float(lam),
                    iterations=iteration,
                    residual_norm=residual_norm,
                    displacement_norm=float(np.linalg.norm(u)),
                    max_equivalent_plastic_strain=ns._max_plastic_strain(committed_states),
                    control_value=current,
                    active_stage=active_stage,
                    support_reactions=support_reactions,
                )
            )
            history.append(
                {
                    "step_index": step_index,
                    "load_factor": float(lam),
                    "control_value": current,
                    "target_displacement": target,
                    "residual_norm": residual_norm,
                    "constraint_error": constraint_error,
                    "iterations": iteration,
                    "active_stage": active_stage,
                    "linearization": "block_elimination",
                    "support_reactions": {
                        name: list(values)
                        for name, values in support_reactions.items()
                    },
                }
            )
            if record_increment_snapshots:
                snapshots.append(
                    ns._increment_snapshot(
                        step_index,
                        lam,
                        u,
                        committed_states,
                        control_value=current,
                    )
                )
                exact_guard(
                    model,
                    context="nonlinear displacement-control block snapshot",
                )
            if progress_callback is not None:
                ns.emit_progress(
                    progress_callback,
                    "nonlinear_static_step",
                    "nonlinear_static.displacement.block",
                    completed=step_index,
                    total=num_steps,
                    iteration=iteration,
                    control="displacement",
                    step_index=int(step_index),
                    load_factor=float(lam),
                    control_value=float(current),
                    displacement_norm=float(np.linalg.norm(u)),
                    max_equivalent_plastic_strain=float(
                        ns._max_plastic_strain(committed_states)
                    ),
                    support_reactions={
                        name: list(values)
                        for name, values in support_reactions.items()
                    },
                )
                exact_guard(
                    model,
                    context="nonlinear displacement-control block progress",
                )

    u_final = np.asarray(T @ q + u0, dtype=float).reshape(-1)
    committed_states = ns._materialize_final_nonlinear_states(
        committed_states,
        info,
    )
    exact_guard(
        model,
        context="nonlinear displacement-control block materialization",
    )
    committed_states = ns._finalize_nonlinear_element_states(
        model,
        u_final,
        committed_states,
        num_layers,
        kinematics=kinematics,
    )
    exact_guard(
        model,
        context="nonlinear displacement-control block finalization",
    )
    info["failure_reason"] = failure_reason
    info["stop_reason"] = (
        "target_displacement_reached"
        if failure_reason is None
        else failure_reason
    )
    info["status_category"] = ns._nonlinear_status_category(
        status,
        failure_reason,
    )
    info["last_converged_load_factor"] = float(lam)
    info["peak_load_factor"] = max((step.load_factor for step in steps), default=float(lam))
    info["force_displacement_history"] = history
    info["strain_summary"] = ns._nonlinear_state_summary(committed_states)
    info["total_newton_iterations"] = total_iterations
    info["displacement_control_linearization"] = "block_elimination"
    info["reaction_force_recovery"] = {
        "accepted_force_reuse_count": reaction_force_reuse_count,
        "full_reassembly_count": reaction_force_reassembly_count,
    }
    info["solve_time"] = time.time() - start_time
    info["constraint_postcheck"] = ns.constraint_residual_summary(model, u_final)
    exact_guard(
        model,
        context="nonlinear displacement-control block constraint postcheck",
    )
    info["result_case"] = make_result_case(
        name="nonlinear_static_displacement_control",
        analysis_type="nonlinear_static",
        load_cases=tuple(stage.load_case for stage in load_program.stages) if load_program is not None else (),
        assembly_info={"load": {"vector_type": "load_program" if load_program is not None else "load"}, **info},
        solver_info={"convergence_info": {"status": status}},
        recovery={"displacements": True, "element_states": True, "force_displacement_history": True},
        settings={
            "control": "displacement",
            "num_steps": num_steps,
            "num_layers": num_layers,
            "linearization": "block_elimination",
            "kinematics": kinematics,
            "corotational_tangent": corotational_tangent,
        },
    ).to_dict()
    result = ns.NonlinearStaticResult(
        steps,
        status,
        u_final,
        float(lam),
        committed_states,
        info,
        tuple(snapshots),
    )
    exact_guard(
        model,
        context="nonlinear displacement-control block output",
    )
    return result


def install_nonlinear_performance_optimizations() -> bool:
    """Install the optimized assembly and continuation helpers.

    Returns ``True`` when the performance layer is active and ``False`` when it
    is disabled through ``FE_SOLVER_DISABLE_FAST_NL``.
    """
    global _INSTALLED, _ORIGINAL_ASSEMBLER, _ORIGINAL_SPARSITY_GETTER, _ORIGINAL_DISPLACEMENT_SOLVER
    if os.environ.get("FE_SOLVER_DISABLE_FAST_NL", "").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    if _INSTALLED:
        return True

    from . import matrix_assembly
    from . import nonlinear_static

    _ORIGINAL_ASSEMBLER = nonlinear_static._assemble_nonlinear_system
    _ORIGINAL_SPARSITY_GETTER = matrix_assembly._get_cached_sparsity_pattern
    _ORIGINAL_DISPLACEMENT_SOLVER = nonlinear_static._solve_static_displacement_control
    nonlinear_static._assemble_nonlinear_system = _optimized_assemble_nonlinear_system
    nonlinear_static._solve_static_displacement_control = _solve_static_displacement_control_block
    matrix_assembly._get_cached_sparsity_pattern = _revision_cached_sparsity_pattern
    _INSTALLED = True
    return True


def uninstall_nonlinear_performance_optimizations() -> None:
    """Restore legacy functions, primarily for A/B benchmark tests."""
    global _INSTALLED
    if not _INSTALLED:
        return
    from . import matrix_assembly
    from . import nonlinear_static

    if _ORIGINAL_ASSEMBLER is not None:
        nonlinear_static._assemble_nonlinear_system = _ORIGINAL_ASSEMBLER
    if _ORIGINAL_DISPLACEMENT_SOLVER is not None:
        nonlinear_static._solve_static_displacement_control = _ORIGINAL_DISPLACEMENT_SOLVER
    if _ORIGINAL_SPARSITY_GETTER is not None:
        matrix_assembly._get_cached_sparsity_pattern = _ORIGINAL_SPARSITY_GETTER
    _INSTALLED = False
    clear_nonlinear_assembly_cache()


def nonlinear_performance_status() -> Dict[str, Any]:
    return {
        "installed": bool(_INSTALLED),
        "disabled_by_environment": os.environ.get("FE_SOLVER_DISABLE_FAST_NL", "").strip().lower()
        in {"1", "true", "yes", "on"},
        "cached_models": len(_PLAN_CACHE),
        "diagnostics": nonlinear_assembly_diagnostics(),
    }
