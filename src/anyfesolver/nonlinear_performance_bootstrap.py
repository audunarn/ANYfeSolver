"""Bootstrap nonlinear performance batches without requiring hashable FEModel objects."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import threading
import weakref
from typing import Any, Dict, Optional

from . import corotational_performance as _corotational
from . import linalg as _linalg
from . import nonlinear_performance as _performance
from . import nonlinear_performance_batch_b as _batch_b
from . import nonlinear_performance_batch_c as _batch_c
from .jit_compiler import JIT_DISABLED_REASON, JIT_ENABLED

# Keep one size-aware backend policy in every environment. AutoSparseSolverBackend
# already falls back to SuperLU when PyPardiso is unavailable, while also
# recording the same provenance metadata used when PyPardiso is installed.
if not isinstance(_linalg.DEFAULT_BACKEND, _linalg.AutoSparseSolverBackend):
    _linalg.DEFAULT_BACKEND = _linalg.AutoSparseSolverBackend()

MAX_NONLINEAR_LAYER_PLANS_PER_MODEL = 8

_CACHE_LOCK = threading.RLock()
_INSTALL_LOCK = threading.RLock()


@dataclass
class _ModelPlanCache:
    model_ref: weakref.ReferenceType[Any]
    plans: OrderedDict[int, Any]
    evictions: int = 0


# FEModel is a mutable dataclass and therefore intentionally unhashable. Cache
# by object identity and keep a weak reference so entries disappear naturally.
_PLAN_CACHE: Dict[int, _ModelPlanCache] = {}
_PLAN_CACHE_EVICTIONS = 0


def _purge_dead(model_id: int) -> None:
    with _CACHE_LOCK:
        _PLAN_CACHE.pop(int(model_id), None)


def get_nonlinear_assembly_plan(model: Any, num_layers: int):
    global _PLAN_CACHE_EVICTIONS

    model_id = id(model)
    layers = int(num_layers)
    with _CACHE_LOCK:
        entry = _PLAN_CACHE.get(model_id)
        if entry is None or entry.model_ref() is not model:
            model_ref = weakref.ref(model, lambda _ref, key=model_id: _purge_dead(key))
            entry = _ModelPlanCache(model_ref=model_ref, plans=OrderedDict())
            _PLAN_CACHE[model_id] = entry
        plan = entry.plans.get(layers)
        if plan is None or not plan.is_valid(model, layers):
            plan = _performance.NonlinearAssemblyPlan.build(model, layers)
            if layers not in entry.plans and len(entry.plans) >= MAX_NONLINEAR_LAYER_PLANS_PER_MODEL:
                entry.plans.popitem(last=False)
                entry.evictions += 1
                _PLAN_CACHE_EVICTIONS += 1
            entry.plans[layers] = plan
        entry.plans.move_to_end(layers)
        return plan


def clear_nonlinear_assembly_cache(model: Optional[Any] = None) -> None:
    global _PLAN_CACHE_EVICTIONS

    with _CACHE_LOCK:
        if model is None:
            _PLAN_CACHE.clear()
            _PLAN_CACHE_EVICTIONS = 0
        else:
            _PLAN_CACHE.pop(id(model), None)


def nonlinear_assembly_diagnostics(model: Optional[Any] = None) -> Dict[str, Any]:
    with _CACHE_LOCK:
        if model is not None:
            entry = _PLAN_CACHE.get(id(model))
            if entry is None or entry.model_ref() is not model:
                return {}
            return {
                str(layers): plan.diagnostics()
                for layers, plan in entry.plans.items()
            }
        result: Dict[str, Any] = {}
        for model_id, entry in list(_PLAN_CACHE.items()):
            cached_model = entry.model_ref()
            if cached_model is None:
                continue
            result[str(model_id)] = {
                "model_name": getattr(cached_model, "name", None),
                "plans": {
                    str(layers): plan.diagnostics()
                    for layers, plan in entry.plans.items()
                },
                "max_layer_plans": MAX_NONLINEAR_LAYER_PLANS_PER_MODEL,
                "evictions": int(entry.evictions),
            }
        return result


_BASE_INSTALL = _performance.install_nonlinear_performance_optimizations
_BASE_UNINSTALL = _performance.uninstall_nonlinear_performance_optimizations


def install_nonlinear_performance_optimizations() -> bool:
    with _INSTALL_LOCK:
        active = bool(_BASE_INSTALL())
        if active and JIT_ENABLED:
            _batch_b.install_batch_b_optimizations()
            _batch_c.install_batch_c_optimizations()
        return active


def uninstall_nonlinear_performance_optimizations() -> None:
    with _INSTALL_LOCK:
        _batch_c.uninstall_batch_c_optimizations()
        _batch_b.uninstall_batch_b_optimizations()
        _BASE_UNINSTALL()


def nonlinear_performance_status() -> Dict[str, Any]:
    with _INSTALL_LOCK:
        batch_b = _batch_b.batch_b_status()
        batch_b["eligible"] = bool(JIT_ENABLED)
        batch_b["disabled_reason"] = None if JIT_ENABLED else JIT_DISABLED_REASON
        batch_c = _batch_c.batch_c_status()
        batch_c["eligible"] = bool(JIT_ENABLED)
        batch_c["disabled_reason"] = None if JIT_ENABLED else JIT_DISABLED_REASON
        with _CACHE_LOCK:
            cached_models = len(_PLAN_CACHE)
            eviction_count = int(_PLAN_CACHE_EVICTIONS)
        return {
            "installed": bool(_performance._INSTALLED),
            "batch_b": batch_b,
            "batch_c": batch_c,
            "corotational": _corotational.corotational_performance_status(),
            "cached_models": cached_models,
            "cache_policy": {
                "max_layer_plans_per_model": MAX_NONLINEAR_LAYER_PLANS_PER_MODEL,
                "layer_plan_evictions": eviction_count,
            },
            "diagnostics": nonlinear_assembly_diagnostics(),
        }


# Replace the initial WeakKeyDictionary helpers before installation. Functions
# in nonlinear_performance resolve these names dynamically, so the optimized
# assembler immediately uses the identity/weak-reference cache above.
_performance.get_nonlinear_assembly_plan = get_nonlinear_assembly_plan
_performance.clear_nonlinear_assembly_cache = clear_nonlinear_assembly_cache
_performance.nonlinear_assembly_diagnostics = nonlinear_assembly_diagnostics
_performance.nonlinear_performance_status = nonlinear_performance_status
_performance.install_nonlinear_performance_optimizations = install_nonlinear_performance_optimizations
_performance.uninstall_nonlinear_performance_optimizations = uninstall_nonlinear_performance_optimizations

NonlinearAssemblyPlan = _performance.NonlinearAssemblyPlan
