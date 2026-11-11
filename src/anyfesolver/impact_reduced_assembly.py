"""Conservative direct reduced-coordinate assembly for nonlinear impact.

The nonlinear impact solver has more stateful features than the static Batch C
path (HHT history, contact iteration, damage and deletion).  This module keeps
the optimization behind a small eligibility/controller seam so unsupported
physics continues through the full-coordinate oracle without changing public
state semantics.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from scipy import sparse

from .jit_compiler import JIT_DISABLED_REASON, JIT_ENABLED
from .nonlinear_reduced_assembly import (
    ReducedAssemblyPlan,
    ReducedAssemblyPlanLimit,
    assemble_reduced_system,
    build_reduced_assembly_plan,
)


_DEFAULT_MIN_ESTIMATED_ASSEMBLIES = 144


def _minimum_estimated_assemblies() -> int:
    """Use the same measured break-even gate as nonlinear Batch C."""

    raw = os.environ.get(
        "FE_SOLVER_BATCH_C_MIN_ESTIMATED_ASSEMBLIES",
        str(_DEFAULT_MIN_ESTIMATED_ASSEMBLIES),
    )
    try:
        return max(int(raw), 0)
    except ValueError:
        return _DEFAULT_MIN_ESTIMATED_ASSEMBLIES


def estimate_impact_assemblies(num_steps: int, max_iterations: int) -> int:
    """Estimate useful nonlinear assemblies without assuming failed Newton loops."""

    typical_per_step = max(1, min(int(max_iterations), 4))
    return max(int(num_steps), 0) * typical_per_step


def _identity_transformation(transformation: sparse.spmatrix) -> bool:
    transformation = sparse.csr_matrix(transformation, dtype=float)
    transformation.sum_duplicates()
    transformation.eliminate_zeros()
    transformation.sort_indices()
    if (
        transformation.shape[0] != transformation.shape[1]
        or transformation.nnz != transformation.shape[0]
    ):
        return False
    return bool(
        np.array_equal(
            transformation.indptr,
            np.arange(transformation.shape[0] + 1, dtype=transformation.indptr.dtype),
        )
        and np.array_equal(
            transformation.indices,
            np.arange(transformation.shape[0], dtype=transformation.indices.dtype),
        )
        and np.all(transformation.data == 1.0)
    )


def _has_unqualified_material_history(model: Any) -> bool:
    """Return whether the model can create ordinary constitutive history.

    The direct-reduced impact path is currently qualified only for elastic
    element response.  A material hardening curve activates isotropic J2
    history, while a Hill yield surface activates orthotropic plastic history;
    either must therefore stay on the full-coordinate oracle path even when no
    separate impact-damage configuration was requested.
    """

    materials = getattr(model, "materials", None)
    if not isinstance(materials, Mapping):
        return False
    return any(
        getattr(material, "hardening_curve", None) is not None
        or getattr(material, "hill_yield", None) is not None
        for material in materials.values()
    )


def _has_unqualified_beam_fiber_history(model: Any) -> bool:
    """Detect element-owned beam fiber plasticity without validating config."""

    mesh = getattr(model, "mesh", None)
    elements = getattr(mesh, "elements", None)
    if not isinstance(elements, Mapping):
        return False
    for element in elements.values():
        marker = getattr(element, "_fiber_plasticity", None)
        if marker is not None:
            return True
        cross_section = getattr(element, "cross_section", None)
        if (
            isinstance(cross_section, Mapping)
            and cross_section.get("fiber_plasticity") is not None
        ):
            return True
    return False


def _has_native_total_lagrangian_elements(model: Any) -> bool:
    elements = getattr(getattr(model, "mesh", None), "elements", None)
    return bool(
        isinstance(elements, Mapping)
        and any(
            bool(getattr(element, "formulation_native_total_lagrangian", False))
            for element in elements.values()
        )
    )


@dataclass
class ImpactReducedAssemblyController:
    """Prepared direct-reduction path plus stable diagnostics and counters."""

    estimated_assemblies: int
    activation_threshold: int
    exclusion_reasons: Tuple[str, ...]
    fallback_detail: Optional[str] = None
    nonlinear_plan: Optional[Any] = None
    reduced_plan: Optional[ReducedAssemblyPlan] = None
    direct_reduced_assembly_count: int = 0
    direct_reduced_tangent_assembly_count: int = 0
    direct_reduced_residual_assembly_count: int = 0

    @property
    def active(self) -> bool:
        return self.nonlinear_plan is not None and self.reduced_plan is not None

    @property
    def fallback_reason(self) -> Optional[str]:
        return self.exclusion_reasons[0] if self.exclusion_reasons else None

    def assemble(
        self,
        displacements: np.ndarray,
        committed_states: Mapping[int, Any],
        *,
        tangent: bool,
    ):
        if not self.active:
            raise RuntimeError("direct reduced impact assembly is not active")
        force, stiffness, trial_states = assemble_reduced_system(
            self.nonlinear_plan,
            self.reduced_plan,
            displacements,
            committed_states,
            tangent=tangent,
        )
        self.direct_reduced_assembly_count += 1
        if tangent:
            self.direct_reduced_tangent_assembly_count += 1
        else:
            self.direct_reduced_residual_assembly_count += 1
        return force, stiffness, trial_states

    def diagnostics(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "path": (
                "direct_reduced_coordinate_assembly"
                if self.active
                else "full_coordinate_assembly_then_projection"
            ),
            "activated": bool(self.active),
            "estimated_assemblies": int(self.estimated_assemblies),
            "activation_threshold": int(self.activation_threshold),
            "fallback_reason": self.fallback_reason,
            "fallback_detail": self.fallback_detail,
            "exclusion_reasons": list(self.exclusion_reasons),
            "direct_reduced_assembly_count": int(self.direct_reduced_assembly_count),
            "direct_reduced_tangent_assembly_count": int(
                self.direct_reduced_tangent_assembly_count
            ),
            "direct_reduced_residual_assembly_count": int(
                self.direct_reduced_residual_assembly_count
            ),
        }
        if self.reduced_plan is not None:
            result["plan"] = self.reduced_plan.diagnostics()
        return result


def prepare_impact_reduced_assembly(
    model: Any,
    transformation: sparse.spmatrix,
    affine_offset: np.ndarray,
    *,
    num_layers: int,
    kinematics: str,
    plastic_damage_enabled: bool,
    num_steps: int,
    max_iterations: int,
) -> ImpactReducedAssemblyController:
    """Prepare direct assembly or return an observable conservative fallback."""

    estimated = estimate_impact_assemblies(num_steps, max_iterations)
    threshold = _minimum_estimated_assemblies()
    exclusions = []
    if not JIT_ENABLED:
        exclusions.append("jit_unavailable")
    if str(kinematics).lower() != "von_karman":
        exclusions.append("unsupported_kinematics")
    if bool(plastic_damage_enabled):
        exclusions.append("plastic_damage_or_erosion_enabled")
    if _has_unqualified_material_history(model):
        exclusions.append("plastic_material_history_unqualified")
    if _has_unqualified_beam_fiber_history(model):
        exclusions.append("beam_fiber_plasticity_unqualified")
    if _has_native_total_lagrangian_elements(model):
        exclusions.append("native_total_lagrangian_state_requires_full_coordinates")
    if np.any(np.asarray(affine_offset, dtype=float) != 0.0):
        exclusions.append("affine_constraint_offset_nonzero")
    if transformation.shape[1] == 0:
        exclusions.append("empty_reduced_system")
    elif _identity_transformation(transformation):
        exclusions.append("identity_constraint_transformation")
    if estimated < threshold:
        exclusions.append("estimated_assembly_budget_below_threshold")

    if exclusions:
        detail = JIT_DISABLED_REASON if "jit_unavailable" in exclusions else None
        return ImpactReducedAssemblyController(
            estimated_assemblies=estimated,
            activation_threshold=threshold,
            exclusion_reasons=tuple(exclusions),
            fallback_detail=detail,
        )

    try:
        # Installation retains Batch B's in-place elastic shell kernel for the
        # local-response half of direct reduced assembly.  It is idempotent.
        from .nonlinear_performance_bootstrap import (
            get_nonlinear_assembly_plan,
            install_nonlinear_performance_optimizations,
        )

        if not install_nonlinear_performance_optimizations():
            return ImpactReducedAssemblyController(
                estimated_assemblies=estimated,
                activation_threshold=threshold,
                exclusion_reasons=("nonlinear_plan_unavailable",),
                fallback_detail="nonlinear performance installation was unavailable",
            )
        nonlinear_plan = get_nonlinear_assembly_plan(model, int(num_layers))
    except Exception as exc:
        return ImpactReducedAssemblyController(
            estimated_assemblies=estimated,
            activation_threshold=threshold,
            exclusion_reasons=("nonlinear_plan_unavailable",),
            fallback_detail=f"{type(exc).__name__}: {exc}",
        )

    try:
        reduced_plan = build_reduced_assembly_plan(
            nonlinear_plan,
            transformation,
        )
    except ReducedAssemblyPlanLimit as exc:
        return ImpactReducedAssemblyController(
            estimated_assemblies=estimated,
            activation_threshold=threshold,
            exclusion_reasons=("reduced_map_memory_limit",),
            fallback_detail=str(exc),
            nonlinear_plan=nonlinear_plan,
        )
    except Exception as exc:
        return ImpactReducedAssemblyController(
            estimated_assemblies=estimated,
            activation_threshold=threshold,
            exclusion_reasons=("reduced_plan_build_failed",),
            fallback_detail=f"{type(exc).__name__}: {exc}",
            nonlinear_plan=nonlinear_plan,
        )

    return ImpactReducedAssemblyController(
        estimated_assemblies=estimated,
        activation_threshold=threshold,
        exclusion_reasons=(),
        nonlinear_plan=nonlinear_plan,
        reduced_plan=reduced_plan,
    )
