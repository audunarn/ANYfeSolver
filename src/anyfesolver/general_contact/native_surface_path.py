"""Bounded persistence certificate for an explicitly prescribed material path.

This certifies neither first touch nor release. Unresolved intervals require
subdivision by the future analysis owner; endpoint history must not be committed.
"""
from dataclasses import dataclass
from fractions import Fraction
from collections import OrderedDict

import numpy as np

from anyfesolver._ge_beam3_p5.compensated_coordinates import advance
from anyfesolver._ge_beam3_p5.algebra import rotation
from anyfesolver._ge_beam3_p5_seeded.core import sha
from .native_surface_s3_friction import NativeS3MaterialConfiguration


# Geometry only: no force, material state, certificate or analysis deadline is
# cached. Exact binary64 inputs include every shape. At most 64 small entries
# retain less than 1 MiB of input/output array payload; larger paths bypass it.
_PATH_POSES = OrderedDict()


def _path_pose_key(start, translations, angular, shell_increment, fraction):
    arrays = (start.positions, start.position_low, start.cell_rotations,
              start.shell_displacement, translations, angular, shell_increment)
    if sum(value.nbytes for value in arrays) > 8192:
        return None
    return tuple((value.shape, value.dtype.str, value.tobytes()) for value in arrays) + (
        float(fraction).hex(),)


def _immutable_path_pose(pose):
    # A bytes backing store cannot be made writable by a caller. Sharing this
    # already validated geometry therefore cannot leak mutations across paths.
    result = object.__new__(NativeS3MaterialConfiguration)
    for name in ('positions', 'position_low', 'cell_rotations', 'shell_displacement'):
        value = getattr(pose, name)
        object.__setattr__(result, name,
            np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape))
    return result


@dataclass(frozen=True)
class MaterialPathCertificate:
    certified: bool
    reason: str
    intervals: int
    path_sha256: str


@dataclass(frozen=True)
class AffineMaterialContactEvent:
    """Geometric event proposal, not an equilibrated/committable increment."""
    fraction: float
    transition: str
    pose: NativeS3MaterialConfiguration
    path_sha256: str
    root_gap_m: float
    root_roundoff_bound_m: float


def affine_gap_polynomial(pair, start, translations, angular, shell_increment):
    """Exact oriented-volume coefficients for the represented affine path.

    Binary64 corners, rotated arm and increments define this path. Native base
    interpolation retains both coordinate parts. Normalization is unnecessary
    for signs/roots. This is an exact predicate, not a penetration tolerance.
    Returns None when rotation or target deformation makes the path non-affine.
    """
    moved = np.asarray(pair.target.adapter.work_map @ np.asarray(shell_increment)).reshape(3, 3)
    if np.any(angular) or not np.array_equal(moved, np.tile(moved[0], (3, 1))):
        return None
    F = lambda value: Fraction.from_float(float(value))
    corners = pair.target.adapter.current_positions(start.shell_displacement)
    edge1 = [F(corners[1,i])-F(corners[0,i]) for i in range(3)]
    edge2 = [F(corners[2,i])-F(corners[0,i]) for i in range(3)]
    normal = [edge1[1]*edge2[2]-edge1[2]*edge2[1],
              edge1[2]*edge2[0]-edge1[0]*edge2[2],
              edge1[0]*edge2[1]-edge1[1]*edge2[0]]
    source = pair.source; cell = source.cell; t = F(source.t)
    station = source.evaluate(start.positions, start.position_low, start.cell_rotations)
    arm = start.cell_rotations[source.cell] @ (source._reference_lift + source._radial_reference)
    base = [(1-t)*(F(start.positions[cell,i])+F(start.position_low[cell,i]))
            +t*(F(start.positions[cell+1,i])+F(start.position_low[cell+1,i]))
            +F(arm[i])-F(corners[0,i]) for i in range(3)]
    velocity = [(1-t)*F(translations[cell,i])+t*F(translations[cell+1,i])-F(moved[0,i])
                for i in range(3)]
    return sum(n*x for n,x in zip(normal,base)), sum(n*v for n,v in zip(normal,velocity))


def material_path_pose(start, translations, angular, shell_increment, fraction):
    """Affine translations and declared spatial Exp(alpha*omega) rotations.

    Angular increments are supplied, never inferred from endpoint rotations.
    Compensated coordinates retain their native canonical representation.
    """
    translations = np.asarray(translations, dtype=float)
    angular = np.asarray(angular, dtype=float)
    shell_increment = np.asarray(shell_increment, dtype=float)
    if (translations.shape != (3, 3) or angular.shape != (2, 3)
            or shell_increment.shape != start.shell_displacement.shape
            or not all(np.all(np.isfinite(v)) for v in
                       (translations, angular, shell_increment))
            or not np.isfinite(fraction) or not 0 <= fraction <= 1):
        raise ValueError('finite native/S3 path increments and fraction required')
    if np.max(np.linalg.norm(angular, axis=1)) >= .9*np.pi:
        raise ValueError('native material path exceeds admitted rotation chart')
    key = _path_pose_key(start, translations, angular, shell_increment, fraction)
    if key is not None and key in _PATH_POSES:
        _PATH_POSES.move_to_end(key)
        return _PATH_POSES[key]
    high = np.empty((3, 3)); low = np.empty((3, 3))
    for index in np.ndindex(3, 3):
        high[index], low[index] = advance(float(start.positions[index]),
            float(start.position_low[index]), float(fraction*translations[index]))
    cells = np.array([rotation(fraction*angular[i]) @ start.cell_rotations[i] for i in range(2)])
    pose = NativeS3MaterialConfiguration(high, low, cells,
        start.shell_displacement + fraction*shell_increment)
    if key is not None:
        pose = _immutable_path_pose(pose)
        _PATH_POSES[key] = pose
        if len(_PATH_POSES) > 64:
            _PATH_POSES.popitem(last=False)
    return pose


def certify_material_path(pair, start, translations, angular, shell_increment,
                          *, closed, maximum_intervals=256):
    if type(closed) is not bool:
        raise ValueError('explicit persistent contact state required')
    return _certify_material_path(pair, start, translations, angular, shell_increment,
                                  closed=closed, maximum_intervals=maximum_intervals)


def _ownership_motion_identity(start,translations,angular,shell_increment,maximum_intervals):
    return sha(dict(positions=start.positions.tolist(),position_low=start.position_low.tolist(),
        rotations=start.cell_rotations.tolist(),shell=start.shell_displacement.tolist(),
        translations=np.asarray(translations).tolist(),angular=np.asarray(angular).tolist(),
        shell_increment=np.asarray(shell_increment).tolist(),maximum_intervals=maximum_intervals))


def _ownership_path_identity(pair,start,translations,angular,shell_increment,maximum_intervals):
    return sha(dict(definition=pair.definition_sha256,motion=_ownership_motion_identity(
        start,translations,angular,shell_increment,maximum_intervals)))


def certify_material_ownership(pair, start, translations, angular, shell_increment,
                               *, maximum_intervals=256):
    """Certify only physical facet ownership; this does not admit contact history."""
    prepared=getattr(pair,'_prepared_history_ownership',None)
    if prepared is not None:
        if sha(pair._definition())!=pair.definition_sha256:
            raise ValueError('native/S3 material-point definition changed')
        identity=_ownership_path_identity(pair,start,translations,angular,shell_increment,maximum_intervals)
        if identity==prepared[0]:
            # Issued only by a field whose complete-domain temporal ownership
            # certificate passed in this evaluation. No gap or event admission
            # follows from this proof, and a changed path cannot reuse it.
            return MaterialPathCertificate(True,'bound whole-history target ownership',0,
                sha(dict(path=identity,whole_history=prepared[1])))
    return _certify_material_path(pair, start, translations, angular, shell_increment,
                                  closed=None, maximum_intervals=maximum_intervals)


def _certify_material_path(pair, start, translations, angular, shell_increment,
                           *, closed, maximum_intervals):
    """Certify strict facet ownership and unchanged open/closed state.

    Quadratic Bernstein bounds certify target orientation and area. Global
    Lipschitz bounds enclose gap and each signed physical edge distance. A
    second-derivative interpolation bound also encloses the gap on each interval.
    Sampling detects failures but never establishes success.
    """
    if type(maximum_intervals) is not int or maximum_intervals < 1:
        raise ValueError('explicit state and positive interval cap required')
    if sha(pair._definition()) != pair.definition_sha256:
        raise ValueError('native/S3 material-point definition changed')
    end = material_path_pose(start, translations, angular, shell_increment, 1.)
    identity = sha(dict(profile='NATIVE_S3_PERSISTENT_PATH_DEV_V1',
        definition=pair.definition_sha256, positions=start.positions.tolist(),
        position_low=start.position_low.tolist(), rotations=start.cell_rotations.tolist(),
        shell=start.shell_displacement.tolist(), translations=np.asarray(translations).tolist(),
        angular=np.asarray(angular).tolist(), shell_increment=np.asarray(shell_increment).tolist(),
        closed=closed, maximum_intervals=maximum_intervals))

    def result(certified, reason, intervals):
        return MaterialPathCertificate(certified, reason, intervals, identity)

    source, target = pair.source, pair.target
    first = target.adapter.current_positions(start.shell_displacement)
    last = target.adapter.current_positions(end.shell_displacement)
    target.physical_pose(start.shell_displacement)
    target.physical_pose(end.shell_displacement)
    a0, b0 = first[1]-first[0], first[2]-first[0]
    a1, b1 = last[1]-last[0], last[2]-last[0]
    c0, c2 = np.cross(a0, b0), np.cross(a1, b1)
    c1 = .5*(np.cross(a0, b1)+np.cross(a1, b0))
    reference = target.adapter.reference_positions
    ref = np.cross(reference[1]-reference[0], reference[2]-reference[0])
    ref_area = np.linalg.norm(ref); ref /= ref_area
    area_lower = min(float(ref @ c) for c in (c0, c1, c2))
    area_scale = max(ref_area, max(np.linalg.norm(a0), np.linalg.norm(a1))
                     *max(np.linalg.norm(b0), np.linalg.norm(b1)))
    if area_lower <= 1e-12*area_scale:
        return result(False, 'target orientation/area unresolved', 0)
    normal_speed = (np.linalg.norm(a1-a0)*max(np.linalg.norm(b0), np.linalg.norm(b1))
        +np.linalg.norm(b1-b0)*max(np.linalg.norm(a0), np.linalg.norm(a1)))/area_lower
    da,db=a1-a0,b1-b0
    cross_speed=max(np.linalg.norm(np.cross(da,b0)+np.cross(a0,db)),
                    np.linalg.norm(np.cross(da,b1)+np.cross(a1,db)))
    normal_acceleration=(2*np.linalg.norm(np.cross(da,db))/area_lower
                         +3*cross_speed**2/area_lower**2)
    source.evaluate(start.positions, start.position_low, start.cell_rotations)
    arm = np.linalg.norm(start.cell_rotations[source.cell] @
                         (source._reference_lift + source._radial_reference))
    nodes0 = start.positions+start.position_low
    nodes1 = end.positions+end.position_low
    cell, t = source.cell, source.t
    base0 = (1-t)*nodes0[cell]+t*nodes0[cell+1]
    base1 = (1-t)*nodes1[cell]+t*nodes1[cell+1]
    speed = np.linalg.norm((1-t)*np.asarray(translations)[cell]+t*np.asarray(translations)[cell+1])
    speed += np.linalg.norm(np.asarray(angular)[cell])*arm
    rates = []
    # The point law requires every barycentric coordinate >1e-12. Each
    # opposite altitude is bounded by the longest affine endpoint edge;
    # subtract that fixed bound so certification implies the same ownership.
    altitude_upper = max(np.linalg.norm(corners[(i+1)%3]-corners[i])
                         for corners in (first, last) for i in range(3))
    for i in range(3):
        radius = max(np.linalg.norm(base0-first[i]), np.linalg.norm(base1-last[i]))+arm
        relative_speed = speed+np.linalg.norm(last[i]-first[i])
        if i == 0:
            gap_rate = relative_speed+normal_speed*radius
            gap_curvature=(normal_acceleration*radius+2*normal_speed*relative_speed
                           +np.linalg.norm(np.asarray(angular)[cell])**2*arm)
        edge = first[(i+1)%3]-first[i]
        change = last[(i+1)%3]-last[i]-edge
        denominator = change @ change
        alpha = 0. if denominator == 0 else np.clip(-(edge @ change)/denominator, 0., 1.)
        lower = np.linalg.norm(edge+alpha*change)
        if lower == 0:
            return result(False, 'target edge unresolved', 0)
        rates.append(relative_speed+(normal_speed+np.linalg.norm(change)/lower)*radius)
    rates = np.array([0. if closed is None else gap_rate, *rates])
    padding = 256*np.finfo(float).eps*max(1., np.max(np.abs(nodes0)),
        np.max(np.abs(nodes1)), np.max(np.abs(first)), np.max(np.abs(last)), arm)

    def margins(alpha):
        pose = material_path_pose(start, translations, angular, shell_increment, alpha)
        corners = target.adapter.current_positions(pose.shell_displacement)
        frame = np.asarray(target.physical_pose(pose.shell_displacement).material_frame)
        normal = frame[:, 2]
        point = source.evaluate(pose.positions, pose.position_low, pose.cell_rotations).surface_position
        gap = float(normal @ (point-corners[0]))
        result = [1. if closed is None else (-gap if closed else gap)]
        for i in range(3):
            edge = corners[(i+1)%3]-corners[i]
            result.append(float(np.cross(normal, edge/np.linalg.norm(edge)) @ (point-corners[i]))
                          -1e-12*altitude_upper)
        return np.asarray(result)

    left, right = margins(0.), margins(1.)
    # With no section rotation and a rigidly translating target, the normal,
    # edge directions, gap and edge margins are affine exactly. This branch
    # admits a zero-gap endpoint without an endless Lipschitz subdivision.
    # Use the work map's increments, not subtraction of rounded endpoints.
    polynomial = affine_gap_polynomial(pair, start, np.asarray(translations), angular, shell_increment)
    if polynomial is not None:
        gap0, change = polynomial
        sign = -1 if closed else 1
        if ((closed is None or min(sign*gap0, sign*(gap0+change)) >= 0)
                and min(np.min(left[1:]), np.min(right[1:])) > padding):
            return result(True, 'affine persistent interior state', 1)
        return result(False, 'affine opening/closure or facet exit', 1)
    stack = [(0., 1., left, right)]; count = 0
    while stack:
        if count == maximum_intervals:
            return result(False, 'interval budget exhausted', count)
        lo, hi, left, right = stack.pop(); count += 1
        if min(np.min(left), np.min(right)) < 0:
            return result(False, 'opening/closure or facet exit', count)
        lower=np.minimum(left,right)-.5*(hi-lo)*rates
        if closed is not None:
            # |g - linear endpoint interpolation| <= sup|g''| h^2/8.
            # The sign-reversed closed margin has the same curvature bound.
            lower[0]=max(lower[0],min(left[0],right[0])-gap_curvature*(hi-lo)**2/8)
        if np.all(lower-padding > 0):
            continue
        mid = .5*(lo+hi)
        if mid in (lo, hi):
            return result(False, 'path resolution exhausted', count)
        middle = margins(mid)
        if np.min(middle) < 0:
            return result(False, 'opening/closure or facet exit', count)
        stack.extend(((lo, mid, left, middle), (mid, hi, middle, right)))
    return result(True, 'persistent interior state', count)


def locate_affine_contact_event(pair, start, translations, angular, shell_increment):
    """Locate one normal transition on a declared exactly affine surface path.

    No rotations or target deformation are admitted here. Interior ownership is
    certified over the whole affine path before returning an event proposal.
    The caller must equilibrate at its continuation parameter before committing.
    General rotational crossing requires a separate bounded event subdivision.
    """
    end = material_path_pose(start, translations, angular, shell_increment, 1.)
    moved = np.asarray(pair.target.adapter.work_map @ np.asarray(shell_increment)).reshape(3, 3)
    if np.any(angular) or not np.array_equal(moved, np.tile(moved[0], (3, 1))):
        raise ValueError('contact event requires affine physical surface motion')
    if sha(pair._definition()) != pair.definition_sha256:
        raise ValueError('native/S3 material-point definition changed')
    first = pair.target.adapter.current_positions(start.shell_displacement)
    frame = np.asarray(pair.target.physical_pose(start.shell_displacement).material_frame)
    normal = frame[:, 2]
    points = [pair.source.evaluate(p.positions, p.position_low, p.cell_rotations).surface_position
              for p in (start, end)]
    last = pair.target.adapter.current_positions(end.shell_displacement)
    # This matches the point law's strict barycentric ownership. On a translating
    # triangle barycentric coordinates are affine, so endpoint checks suffice.
    for point, corners in zip(points, (first, last)):
        uv = np.linalg.lstsq(np.column_stack((corners[1]-corners[0], corners[2]-corners[0])),
                            point-corners[0], rcond=None)[0]
        if min(uv[0], uv[1], 1-uv.sum()) <= 1e-12:
            raise ValueError('contact event path leaves interior ownership')
    gap0, change = affine_gap_polynomial(pair, start, np.asarray(translations), angular, shell_increment)
    gap1 = gap0+change
    if gap0 > 0. and gap1 <= 0.:
        transition = 'first_touch'
    elif gap0 < 0. and gap1 >= 0.:
        transition = 'release'
    elif gap0 == 0. and gap1 != 0.:
        transition = 'first_touch' if gap1 < 0. else 'release'
    else:
        raise ValueError('affine contact path has no unique normal transition')
    alpha = float(0. if gap0 == 0. else gap0/(gap0-gap1))
    pose = material_path_pose(start, translations, angular, shell_increment, alpha)
    # Bind the full approach path, not just its endpoint/zero-gap pose.
    binding = certify_material_path(pair, start, translations, angular, shell_increment,
                                    closed=transition == 'release').path_sha256
    root_point = pair.source.evaluate(pose.positions, pose.position_low, pose.cell_rotations).surface_position
    root_target = pair.target.physical_pose(pose.shell_displacement)
    root_gap = float(normal @ (root_point-np.asarray(root_target.position)))
    bound = 256*np.finfo(float).eps*max(1., np.max(np.abs(root_point)), np.max(np.abs(first)), np.max(np.abs(last)))
    if abs(root_gap) > bound:
        raise ValueError('affine event root failed represented-gap verification')
    return AffineMaterialContactEvent(alpha, transition, pose, binding, root_gap, float(bound))
