"""Development zero-set certificate; not yet wired to material integration.

The caller supplies whole-material-rectangle enclosures of gap and its time
derivative on each slab. Sampling is not an enclosure. Every possible zero
must be transverse with the same orientation. Thus a continuously
differentiable point gap has at most one zero on the complete segment;
turning points strictly away from zero do not obstruct this implication.
"""
from dataclasses import dataclass
from math import isfinite

from .native_surface_support import SupportIntervalBudget


@dataclass(frozen=True)
class GapSlabEnclosure:
    gap_lower: float
    gap_upper: float
    derivative_lower: float
    derivative_upper: float
    spatial_derivative_width: float = 0.
    temporal_derivative_width: float = 0.
    zero_derivative_lower: float | None = None
    zero_derivative_upper: float | None = None

    def __post_init__(self):
        if (not all(isfinite(v) for v in (self.gap_lower,self.gap_upper,
                                         self.derivative_lower,self.derivative_upper))
                or self.gap_lower>self.gap_upper
                or self.derivative_lower>self.derivative_upper):
            raise ValueError('finite ordered whole-slab enclosures required')
        if any(not isfinite(v) or v<0. for v in (self.spatial_derivative_width,self.temporal_derivative_width)):
            raise ValueError('finite nonnegative subdivision widths required')
        if self.zero_derivative_lower is not None or self.zero_derivative_upper is not None:
            if (self.zero_derivative_lower is None or self.zero_derivative_upper is None
                    or not isfinite(self.zero_derivative_lower) or not isfinite(self.zero_derivative_upper)
                    or self.zero_derivative_lower>self.zero_derivative_upper):
                raise ValueError('finite ordered zero-set derivative bounds required')


def certify_single_crossing(enclose, *, budget, allow_spatial_refinement=False):
    """Prove at most one crossing per material point, or return False.

    Bounds must include arithmetic error and the full spatial rectangle.
    The derivative is with respect to the original segment parameter, not
    a rescaled local time. Strict inequalities exclude unresolved tangencies.
    Every inspected slab consumes the caller's shared support budget.
    Optional width hints choose only a non-success return to the spatial
    refiner; they never establish a gap or derivative certificate.
    Optional zero-set bounds enclose the actual derivative at every possible
    gap zero (for example by bounding g_t-k*g with fixed k over the slab).
    They need not bound the derivative away from zero.
    """
    if type(budget) is not SupportIntervalBudget:
        raise TypeError('exact shared support interval budget required')
    pending=[(0.,1.)];orientation=0
    while pending:
        if not budget.consume():
            return False
        left,right=pending.pop()
        bounds=enclose(left,right)
        if type(bounds) is not GapSlabEnclosure:
            raise TypeError('exact whole-slab enclosure required')
        if bounds.gap_lower>0. or bounds.gap_upper<0.:
            continue
        lower=(bounds.derivative_lower if bounds.zero_derivative_lower is None else bounds.zero_derivative_lower)
        upper=(bounds.derivative_upper if bounds.zero_derivative_upper is None else bounds.zero_derivative_upper)
        direction=(1 if lower>0. else -1 if upper<0. else 0)
        if direction:
            if orientation and direction!=orientation:
                return False
            orientation=direction
            continue
        if (allow_spatial_refinement and bounds.spatial_derivative_width>0.
                and bounds.spatial_derivative_width>=bounds.temporal_derivative_width):
            return False
        middle=(left+right)/2
        if not left<middle<right:
            return False
        pending.extend(((middle,right),(left,middle)))
    return True
