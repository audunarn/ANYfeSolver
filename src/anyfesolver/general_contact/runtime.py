"""Lightweight per-analysis contact context for shared nonlinear drivers."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator


_ACTIVE_CONTACT: ContextVar[object | None] = ContextVar(
    "anyfesolver_general_contact", default=None,
)


def active_contact_manager() -> object | None:
    return _ACTIVE_CONTACT.get()


@contextmanager
def suspend_contact() -> Iterator[None]:
    """Keep a nested contact-free solve independent of its calling solve."""

    token = _ACTIVE_CONTACT.set(None)
    try:
        yield
    finally:
        _ACTIVE_CONTACT.reset(token)
