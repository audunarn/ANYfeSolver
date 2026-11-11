"""Explicit bounded larger factor domain; historical defaults are unchanged."""
from contextlib import contextmanager
from contextvars import ContextVar

POLICY='GE_BEAM3_LARGE_EXACT_FACTOR_MODAL_CAPACITY_V1'
_ACTIVE=ContextVar('native_large_factor_modal_capacity',default=False)

def limits():
    return (640,12288,640) if _ACTIVE.get() else (256,8192,512)

def active():return _ACTIVE.get()

@contextmanager
def large_modal_capacity():
    if _ACTIVE.get():raise ValueError('nested large modal capacity forbidden')
    token=_ACTIVE.set(True)
    try:yield
    finally:_ACTIVE.reset(token)

def solve_large_factor_chain_modes(*args,**kwargs):
    from ._native_paired_factor_chain_modes import solve_paired_factor_chain_modes
    with large_modal_capacity():return solve_paired_factor_chain_modes(*args,**kwargs)

def apply_large_mode_map(*args,**kwargs):
    from ._native_paired_factor_chain_modes import apply_mode_map
    with large_modal_capacity():return apply_mode_map(*args,**kwargs)
