"""Explicit guard-work optimization for isolated copied compliance arithmetic."""
from time import monotonic
from .control import cancellation_safe_point
from ._native_reference_modal import _owned
from ._ge_beam3_native_generalized_factor_modal import compliance_factor

POLICY='GE_BEAM3_COMPLIANCE_SNAPSHOT_GUARD_V1'

def deadline_checkpoint(started,cancellation_token=None):
    cancellation_safe_point(cancellation_token,'translation-modal.compliance-snapshot')
    if monotonic()-started>120:raise RuntimeError('retained generalized context deadline')

def snapshot_compliance(value,full_check,checkpoint):
    """No live state is read by inner factor arithmetic; no output escapes unchecked."""
    full_check()
    snapshot=_owned(value)
    try:
        return compliance_factor(snapshot,checkpoint)
    finally:
        full_check()
