"""Correlated final-segment work accounting for certified variable-front strips.

A stationary front strip is the certified axial uncertainty interval of one
isolated root of N=A'B-AB'. When certified B and N' signs make that root the
front's contact-side turning point, material inside the strip penetrates
during the final segment only across the strip width, and its segment-boundary
state is exactly replayable from the prefix history. These are developmental
integral enclosures; they never admit a point history or soften a refusal.
"""
import numpy as np

from .native_surface_history import NativeSurfaceHistoryField, NativeSurfaceReplayBudget


def contact_side_turning(branch_bounds):
    """True when certified B and N' signs make the root the contact-side extreme.

    H=-A/B has H''=-N'/B^2 at a root. For B>0 contact is xi<=H and the
    contact-side extreme is a maximum of H (N'>0); for B<0 contact is xi>=H
    and it is a minimum (N'<0). Strict prepared bounds decide both signs.
    """
    bl, bu, _, _, dl, du = branch_bounds[:6]
    return (bl > 0. and dl > 0.) or (bu < 0. and du < 0.)


def final_segment_strip_work_bound(field, cell, strip_width, segment_index):
    """Final-segment work density for material penetrating only strip_width.

    The bound mirrors the whole-history enclosure's construction, with the
    whole-history sweep depth replaced by the certified strip width. It is
    valid only together with a contact-side turning certificate.
    """
    from .native_surface_front_work import _segment_motion_bounds
    if type(segment_index) is not int or not 0 <= segment_index < len(field.segments):
        raise ValueError('strip work enclosure requires a valid explicit segment index')
    if not np.isfinite(strip_width) or strip_width <= 0.:
        raise ValueError('positive certified strip width required')
    slip, chord, padding = _segment_motion_bounds(field, cell, segment_index)
    pair = field.patch.pairs[0]
    penetration = chord * strip_width + padding
    cap = pair.friction_coefficient * pair.normal_stiffness_Pa_per_m * penetration
    kt = pair.tangential_stiffness_Pa_per_m
    dissipation = cap * (slip + 2. * pair.event_history_substeps * cap / kt)
    storage = cap * cap / (2. * kt)
    result = np.array((storage, dissipation, storage, dissipation))
    if not np.all(np.isfinite(result)) or np.any(result < 0.):
        raise ValueError('nonfinite final-segment strip work enclosure')
    return result


class PrefixReplayField:
    """Immutable prefix-history replay for segment-boundary strip states.

    The truncated field replays the declared path without its final segment,
    so a strip material point whose final-segment touch is tangential can
    still be evaluated exactly at the segment boundary. Construction is lazy
    and cached per integration; the source field is never mutated.
    """
    def __init__(self, field):
        if type(field) is not NativeSurfaceHistoryField:
            raise TypeError('exact material history field required')
        self._field = field
        self._prefix = None

    def field(self):
        if self._prefix is None:
            self._prefix = NativeSurfaceHistoryField(self._field.patch, self._field.initial,
                self._field.segments[:-1], maximum_intervals=self._field.maximum_intervals)
        return self._prefix

    def boundary_state(self, xi, angle, budget):
        """Exact (stored, cumulative dissipation) density at the segment boundary."""
        if type(budget) is not NativeSurfaceReplayBudget:
            raise TypeError('explicit shared material replay budget required')
        value = self.field().query(float(xi), float(angle), budget=budget)
        point = value.last_contribution
        stored = 0. if point is None else float(point.trial.stored_energy_density)
        return stored, float(value.history.cumulative_dissipation_density)
