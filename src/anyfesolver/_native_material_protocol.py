"""Internal opt-in station-state validation for native nonlinear elements.

No mechanics, default routing, or implicit registration lives here. Validators
operate on owned state copies before the store publishes accepted history.
"""

from dataclasses import dataclass, fields
from typing import Any, Callable

import numpy as np


PROTOCOL = "ANY_NATIVE_MATERIAL_VALIDATION_V1"


@dataclass(frozen=True)
class NativeMaterialValidator:
    validate: Callable[..., None]

    def __post_init__(self) -> None:
        if not callable(self.validate):
            raise TypeError("Native material validation requires a callable")


@dataclass(frozen=True)
class NativeMaterialContext:
    """Live solver-owned authority, unlike a detached immutable rotation view."""

    store: Any
    token: Any
    element_id: int
    node_ids: tuple[int, ...]
    reference_directors: np.ndarray

    def require_view(self, view: Any) -> None:
        issued = self.store.native_material_context(self.token, self.element_id)
        if (self.node_ids != issued.node_ids
                or not np.array_equal(self.reference_directors, issued.reference_directors)):
            raise ValueError("Native material context does not match the model binding")
        fresh = self.store.native_element_rotation_view(
            self.token, self.element_id, issued.node_ids, issued.reference_directors
        )
        if type(view) is not type(fresh):
            raise ValueError("Native material context requires its exact rotation view")
        for field in fields(fresh):
            expected = getattr(fresh, field.name)
            actual = getattr(view, field.name)
            equal = (np.array_equal(actual, expected) if isinstance(expected, np.ndarray)
                     else actual == expected)
            if not equal:
                raise ValueError("Detached or mismatched native material rotation view")
