"""Private material-event observations for bounded DC globalization.

These are the existing constitutive branch equations, not alternate forces or
state updates. Unsupported material/state layouts retain the ordinary search.
"""
from dataclasses import dataclass
from collections.abc import Mapping
import numpy as np


@dataclass(frozen=True)
class MaterialEvents:
    values: np.ndarray
    strains: np.ndarray

    @property
    def signature(self):
        return self.values > 0.0


def material_events(model, committed, trial):
    """Copy isotropic DNV layer/fiber loading and hardening indicators.

    Entry margins exactly match the existing return maps. No caller-owned
    arrays or material states are modified, and no trial state is committed.
    """
    from .elements import BeamElement, ShellElement, _elastic_symmetry
    from .material_curves import DNVC208MaterialCurve
    from .plasticity import plane_stress_elastic_matrix

    if not isinstance(trial, Mapping):
        return None
    values, strains = [], []
    for eid, state in sorted(trial.items()):
        if not isinstance(state, Mapping) or "alpha" not in state:
            continue
        old = committed.get(eid)
        if not isinstance(old, Mapping) or not {"alpha", "plastic_strain"} <= old.keys():
            return None
        element = model.mesh.elements[eid]
        material = model.get_material(element.material_name)
        if _elastic_symmetry(material) != "isotropic":
            return None
        curve = getattr(material, "hardening_curve", None)
        if isinstance(element, BeamElement):
            config = element._fiber_plasticity_config(material)
            if config is None or "fiber_strain" not in state:
                return None
            curve = config.material_curve
        elif not isinstance(element, ShellElement) or "layer_strain" not in state:
            return None
        if not isinstance(curve, DNVC208MaterialCurve):
            return None
        alpha_old = np.asarray(old["alpha"], dtype=float)
        flow = np.asarray(curve.flow_stress(alpha_old), dtype=float)
        E = float(material.elastic_modulus)
        if isinstance(element, BeamElement):
            strain = np.asarray(state["fiber_strain"], dtype=float)
            stress = E * (strain - np.asarray(old["plastic_strain"], dtype=float))
            scale = np.maximum(flow, 1.0)
            loading = (np.abs(stress) - (flow + 1.0e-9 * scale)) / scale
        else:
            strain = np.asarray(state["layer_strain"], dtype=float)
            C = plane_stress_elastic_matrix(E, float(material.poisson_ratio))
            stress = (strain - np.asarray(old["plastic_strain"], dtype=float)) @ C.T
            sx, sy, tau = stress.T
            # Preserve the actual return-map comparison's arithmetic order,
            # including its invariant trial basis and entry-margin subtraction.
            a1, a2 = sx + sy, sx - sy
            phi2 = a1**2 / 12.0 + a2**2 / 4.0 + tau**2
            scale = np.maximum(flow**2, 1.0)
            loading = (phi2 - flow**2 / 3.0 - 1.0e-10 * scale) / scale
        values.append(loading.reshape(-1))
        strains.append(strain.reshape(-1))
        for boundary in (curve.eps_p_y1, curve.eps_p_y2):
            values.append(((np.asarray(state["alpha"], dtype=float) - boundary) / boundary).reshape(-1))
    if not values:
        return None
    indicators, all_strains = np.concatenate(values), np.concatenate(strains)
    if not np.all(np.isfinite(indicators)) or not np.all(np.isfinite(all_strains)):
        return None
    return MaterialEvents(indicators.copy(), all_strains.copy())


def crossed_material_event(base, upper, upper_alpha, q, direction, evaluate):
    """Return a resolved post-event trial after an exhausted ordinary search.

    ``evaluate(alpha)`` supplies an uncommitted observation or None for a typed
    domain refusal. A finite bisection locates a branch transition inside the
    last rejected interval. This does not add arbitrary backtracking trials.
    The caller applies its unchanged Armijo rule to the returned trial.
    """
    if base is None or upper is None or not np.isfinite(upper_alpha) or upper_alpha <= 0.0:
        return None
    if base.values.shape != upper.values.shape or base.strains.shape != upper.strains.shape:
        return None
    signature = base.signature
    if np.array_equal(signature, upper.signature):
        return None
    lo, hi = 0.0, float(upper_alpha)
    before, after = base, upper
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if mid == lo or mid == hi:
            break
        trial = evaluate(mid)
        if trial is None or trial["events"] is None:
            return None
        events = trial["events"]
        if events.values.shape != base.values.shape:
            return None
        if np.array_equal(signature, events.signature):
            lo, before = mid, events
        else:
            hi, after = mid, events
        if hi - lo <= 64.0 * np.spacing(hi):
            break
    changed = np.flatnonzero(before.signature != after.signature)
    if not changed.size:
        return None
    eps = np.finfo(float).eps
    slopes = (upper.values - base.values) / upper_alpha
    if np.any(slopes[changed] == 0.0):
        return None
    equation_offset = float(np.max(64.0 * eps / np.abs(slopes[changed])))
    norm_direction = float(np.max(np.abs(direction)))
    if norm_direction == 0.0:
        return None
    state_offset = 64.0 * float(np.max(np.abs(np.spacing(q)))) / norm_direction
    strain_slope = float(np.max(np.abs((upper.strains - base.strains) / upper_alpha)))
    strain_offset = (64.0 * float(np.max(np.abs(np.spacing(base.strains)))) / strain_slope
                     if strain_slope > 0.0 else 0.0)
    event = 0.5 * (lo + hi)
    offset = max(equation_offset, state_offset, strain_offset, 64.0 * np.spacing(event))
    left_alpha, right_alpha = event - offset, event + offset
    if not 0.0 < left_alpha < right_alpha <= upper_alpha:
        return None
    # Crossing must be representable and resolve the selected event. This
    # prevents repeating a zero-length step at a rounded constitutive switch.
    if np.array_equal(q + right_alpha * direction, q):
        return None
    left, right = evaluate(left_alpha), evaluate(right_alpha)
    if left is None or right is None or left["events"] is None or right["events"] is None:
        return None
    if (not np.array_equal(left["events"].signature, signature)
            or np.array_equal(right["events"].signature, signature)):
        return None
    right["alpha"] = right_alpha
    right["event_alpha"] = event
    return right
