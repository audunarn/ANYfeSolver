"""Fail-closed guards for formulation-specific production capabilities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Iterable, Optional


STATEFUL_MATERIAL_RESPONSE_MODE = "stateful_material"
STATELESS_FIXED_GENERALIZED_SECTION_RESPONSE_MODE = (
    "stateless_fixed_generalized_section"
)
_NONLINEAR_MATERIAL_RESPONSE_MODES = frozenset(
    {
        STATEFUL_MATERIAL_RESPONSE_MODE,
        STATELESS_FIXED_GENERALIZED_SECTION_RESPONSE_MODE,
    }
)


class ElementCapabilityError(NotImplementedError):
    """Raised before a workflow evaluates an element with a declared gap."""


def require_model_element_capabilities(
    model: Any,
    capabilities: str | Iterable[str],
    *,
    context: str,
    element_ids: Optional[Iterable[int]] = None,
) -> None:
    """Reject declared element gaps before assembly, state, or geometry work."""

    requested = (
        frozenset((capabilities,))
        if isinstance(capabilities, str)
        else frozenset(str(value) for value in capabilities)
    )
    if not requested:
        raise ValueError("at least one element capability is required")
    blocked: list[
        tuple[int, tuple[str, ...], tuple[tuple[str, str], ...]]
    ] = []
    elements = getattr(getattr(model, "mesh", None), "elements", {})
    selected = (
        None
        if element_ids is None
        else frozenset(int(element_id) for element_id in element_ids)
    )
    for element_id, element in elements.items():
        if selected is not None and int(element_id) not in selected:
            continue
        gaps = frozenset(str(value) for value in getattr(element, "capability_gaps", ()))
        overlap = tuple(sorted(requested & gaps))
        raw_restrictions = getattr(element, "capability_restrictions", {})
        restrictions = (
            {
                str(name): str(disposition)
                for name, disposition in raw_restrictions.items()
            }
            if isinstance(raw_restrictions, Mapping)
            else {}
        )
        restricted = tuple(
            sorted(
                (name, restrictions[name])
                for name in requested
                if name in restrictions
            )
        )
        if overlap or restricted:
            blocked.append((int(element_id), overlap, restricted))
    if not blocked:
        return
    blocked.sort(key=lambda item: item[0])
    def blocked_label(
        item: tuple[int, tuple[str, ...], tuple[tuple[str, str], ...]],
    ) -> str:
        element_id, gaps, restrictions = item
        labels = [*gaps]
        labels.extend(
            f"{name}={disposition}" for name, disposition in restrictions
        )
        return f"{element_id} ({', '.join(labels)})"

    details = "; ".join(blocked_label(item) for item in blocked[:8])
    if len(blocked) > 8:
        details += f"; and {len(blocked) - 8} more"
    has_restriction = any(
        restrictions for _element_id, _gaps, restrictions in blocked
    )
    reason = (
        "element capability disposition rejects the requested profile"
        if has_restriction
        else "element capability PARITY_GAP remains"
    )
    raise ElementCapabilityError(f"{context} is unavailable because {reason}: {details}")


def require_model_nonlinear_workflow_capabilities(
    model: Any,
    *,
    context: str,
) -> None:
    """Guard the geometry-plus-material nonlinear workflow declaratively.

    Every element must close ``nonlinear_geometry``.  Elements declaring a
    stateful material response must additionally close
    ``material_nonlinearity``.  A stateless fixed generalized section is
    intentionally exempt from only that history capability: requesting
    ``material_nonlinearity`` directly remains fail-closed.
    """

    require_model_element_capabilities(
        model,
        "nonlinear_geometry",
        context=context,
    )
    elements = getattr(getattr(model, "mesh", None), "elements", {})
    stateful_ids: list[int] = []
    invalid: list[tuple[int, str]] = []
    for element_id, element in elements.items():
        mode = str(
            getattr(
                element,
                "nonlinear_material_response_mode",
                STATEFUL_MATERIAL_RESPONSE_MODE,
            )
        )
        if mode not in _NONLINEAR_MATERIAL_RESPONSE_MODES:
            invalid.append((int(element_id), mode))
        elif mode == STATEFUL_MATERIAL_RESPONSE_MODE:
            stateful_ids.append(int(element_id))
    if invalid:
        invalid.sort(key=lambda item: item[0])
        details = "; ".join(
            f"{element_id} ({mode!r})" for element_id, mode in invalid[:8]
        )
        if len(invalid) > 8:
            details += f"; and {len(invalid) - 8} more"
        raise ElementCapabilityError(
            f"{context} is unavailable because nonlinear_material_response_mode "
            f"is undeclared or unsupported: {details}"
        )
    require_model_element_capabilities(
        model,
        "material_nonlinearity",
        context=context,
        element_ids=stateful_ids,
    )


__all__ = [
    "ElementCapabilityError",
    "STATEFUL_MATERIAL_RESPONSE_MODE",
    "STATELESS_FIXED_GENERALIZED_SECTION_RESPONSE_MODE",
    "require_model_element_capabilities",
    "require_model_nonlinear_workflow_capabilities",
]
