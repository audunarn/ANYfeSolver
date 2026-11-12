"""Explicit private N32 research capacity; old routes retain their old limits.

No mechanical coefficients, iteration/deadline controls or qualification flags
are changed. The scope must remain active while enlarged contexts are used.
"""
from contextlib import contextmanager
from contextvars import ContextVar

_ACTIVE=ContextVar('ge_beam3_n32_refinement_capacity',default=False)

def retained_limits():
    return (32,1280,640) if _ACTIVE.get() else (24,1024,512)

@contextmanager
def n32_refinement_capacity():
    if _ACTIVE.get():raise ValueError('nested N32 refinement capacity forbidden')
    token=_ACTIVE.set(True)
    try:yield
    finally:_ACTIVE.reset(token)
