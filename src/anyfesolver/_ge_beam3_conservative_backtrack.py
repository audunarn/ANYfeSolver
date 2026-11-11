"""Private conservative backtracking, copied from preserved energy probe.

No final residual relaxation. An indefinite/non-descent force direction
requires an explicit continuation strategy, not silent tangent modification.
"""
import numpy as np

def acceptance(old,new,slope,scale,roundoff_scale,old_norm,new_norm):
    """Armijo c=1e-4; only unresolved energy differences use residual descent.

    The roundoff guard does not relax the final physical residual criterion.
    No stiffness shifts, gradient fallback, step growth or automatic retry.
    """
    if not all(np.isfinite(v) for v in (old,new,slope,scale,roundoff_scale,old_norm,new_norm)):
        raise ValueError('finite line-search quantities required')
    if not (0<scale<=1 and roundoff_scale>=0 and old_norm>=0 and new_norm>=0):
        raise ValueError('valid line-search scales required')
    if slope>=0:
        return 'REJECT_NON_DESCENT'
    floor = 32*np.finfo(float).eps*roundoff_scale
    difference = new-old
    # Resolve near-stationary cancellation before the energy Armijo test:
    # indistinguishable energy alone must never authorize an arbitrary step.
    if abs(difference)<=floor and abs(scale*slope)<=floor:
        return 'ACCEPT_ROUNDOFF_RESIDUAL' if new_norm<old_norm else 'REJECT_ROUNDOFF'
    return 'ACCEPT_ENERGY' if difference<=1e-4*scale*slope else 'REJECT_ENERGY'
