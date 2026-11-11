"""Explicit private validation scheduling for sealed local beam operations.

The default path is unchanged. In the explicit scope, validate the complete
model before and after every local operation (also on error); every existing
inner callback still checks the local sealed operator, deadline and cancellation.
No model/constraint/load data are read by those local operator calculations.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from time import monotonic
from .control import cancellation_safe_point
from ._ge_beam3_effective_deadline import effective_deadline_seconds

POLICY = 'GE_BEAM3_LOCAL_OPERATOR_VALIDATION_BOUNDARY_V1'
_ACTIVE = ContextVar('ge_beam3_local_operator_validation', default=None)


@contextmanager
def operator_validation_scope(*, cancellation_token=None):
    if _ACTIVE.get() is not None:
        raise ValueError('nested operator validation scope forbidden')
    token = _ACTIVE.set((POLICY, cancellation_token))
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def _boundary_call(function, full_check, checkpoint):
    full_check()
    try:
        return function(checkpoint)
    finally:
        full_check()


def operator_call(operator, method, full_check, started, *args, **kwargs):
    if method not in ('evaluate', 'recover') or 'check' in kwargs:
        raise ValueError('registered local operator call required')
    active = _ACTIVE.get()
    if active is None:
        return getattr(operator, method)(*args, check=full_check, **kwargs)
    from ._ge_beam3_retained_generalized import RetainedGeneralizedOperator
    if type(operator) is not RetainedGeneralizedOperator or active[0] != POLICY:
        raise ValueError('exact sealed generalized operator required')
    def checkpoint():
        cancellation_safe_point(active[1], 'ge-beam3.local-operator')
        if monotonic()-started > effective_deadline_seconds(120):
            raise RuntimeError('retained generalized context deadline')
        operator.guard()
    return _boundary_call(
        lambda check: getattr(operator, method)(*args, check=check, **kwargs),
        full_check, checkpoint)
