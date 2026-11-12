"""Opt-in first-approach globalization for native material-area/S3 owners.

Only disposable Newton paths are limited. Geometry queries use the complete
material patch; endpoint samples select a fraction, never certify its path.
The existing radius, director and physical-rim certificates admit the prefix.
"""
import numpy as np
from anyfesolver.control import cancellation_safe_point
from .native_s3_swept import native_s3_swept_admissible


REVISION = 'material_area_counted_first_approach_prefix_v1'


def validate_settings(settings, works, tolerances):
    if settings is None:
        return
    from .native_activation_static import NativeContactActivationSettings
    if type(settings) is not NativeContactActivationSettings:
        raise TypeError('exact native contact activation settings required')
    settings.__post_init__()
    if not max(w.contact.geometry_tolerance for w in works) < settings.activation_gap_m < min(tolerances):
        raise ValueError('activation budget must exceed geometry tolerance and lie inside every pair gap budget')


def activation_fraction(owner, start, u_start, step, fraction, contact, token):
    settings = owner.activation_settings
    if settings is None:
        return fraction
    owner._guard()
    if hasattr(owner, 'pairs'):
        rows = tuple((p.work, i, p.maximum_swept_intervals, r.contact)
                     for (p, r), i in zip(contact.pairs, owner.beam_indices))
    else:
        rows = ((owner.contact, 0, owner.maximum_swept_intervals, contact.contact),)
    opened = tuple(row for row in rows if row[3].minimum_gap_m > 0.)
    if not opened:
        return fraction
    queries = 0

    def query(work, i, pose, u):
        nonlocal queries
        cancellation_safe_point(token, 'native material activation geometry')
        if queries >= settings.maximum_geometry_queries:
            raise RuntimeError('native material activation geometry query budget exhausted')
        queries += 1
        owner.activation_total_queries += 1
        nodes = owner.context.nodes[i]
        return work.geometry_gap(pose.positions[nodes], pose.position_low[nodes],
                                 pose.cell_rotations[i], u)

    def sample(alpha):
        increment = alpha*step
        pose = owner.context.advance(start, increment[:owner.context.count])
        u = u_start+increment[owner.context.count:]
        return pose, u, tuple(query(w, i, pose, u) for w, i, _, _ in opened)

    gap = settings.activation_gap_m
    while True:
        try:
            end, u_end, values = sample(fraction)
            if min(v[0] for v in values) < -.5*gap:
                left, right = 0., fraction
                while True:
                    middle = .5*(left+right)
                    if middle in (left, right):
                        raise RuntimeError('native material activation bracket unresolved')
                    end, u_end, values = sample(middle)
                    minimum = min(v[0] for v in values)
                    if -.75*gap <= minimum <= -.25*gap:
                        fraction = middle
                        break
                    if minimum < -.5*gap:
                        right = middle
                    else:
                        left = middle
            certified = True
            increment = fraction*step
            for (work, i, maximum_intervals, old), (end_gap, end_rim) in zip(opened, values):
                nodes = owner.context.nodes[i]
                pose = work._configuration(start.positions[nodes], start.position_low[nodes],
                                           start.cell_rotations[i], u_start)
                translations = (end.positions[nodes]-start.positions[nodes]
                                +end.position_low[nodes]-start.position_low[nodes])
                angular = np.array([increment[owner.context.nodal_count+24*i+3*c:
                                              owner.context.nodal_count+24*i+3*c+3]
                                    for c in (0, 1)])
                work.certify_orientation_path(pose, translations, angular, u_end-u_start,
                                              maximum_intervals)
                # Leave queries for a shorter prefix when this bound cannot
                # certify. Running out fails; it never omits a physical pair.
                intervals = min(maximum_intervals,
                                max(1, (settings.maximum_geometry_queries-queries)//2))
                if not native_s3_swept_admissible(owner.context, work, i, gap, intervals,
                        owner._recoverable_trial_error, start, u_start, increment, end, u_end,
                        old.minimum_gap_m, end_gap, old.minimum_rim_clearance_m, end_rim,
                        geometry_only=True, projected_gap=True, material_radius=work.contact.radius,
                        geometry_query=lambda state, u, work=work, i=i: query(work, i, state, u)):
                    certified = False
                    break
            if certified:
                return fraction
        except (ValueError, MemoryError) as exc:
            if not owner._recoverable_trial_error(exc):
                raise
        fraction *= .5
        if fraction <= np.finfo(float).eps:
            raise RuntimeError('native material activation prefix fraction unresolved')
