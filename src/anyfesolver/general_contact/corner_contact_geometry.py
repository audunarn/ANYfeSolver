"""Signed closest-point corner gaps on the ORIGINAL bilinear panel faces.

g = s * |x - p| with p the closest point of the union of panel faces and s the
side (positive separated). For an interior projection x - p is along the face
normal and g = n.(x - p). Envelope theorem: dg = u.dx - u.sum_k w_k dX_k with
u = s (x - p)/|x - p| and w the bilinear weights of p; exact first derivative
including edge/vertex points. A shared seam point has identical positions and
edge weights in both faces, so one owner (lowest face index) carries the work and
no force is duplicated. Anything unresolved refuses with ``CornerContactRefusal``.
"""
from __future__ import annotations

import numpy as np

from .corner_contact_certificate import CornerContactRefusal

TIE_M = 1e-12
ZERO_DISTANCE_M = 1e-9


def _shape(xi, eta):
    return 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                            (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])


def _tangents(c, xi, eta):
    dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
    deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
    return dxi @ c, deta @ c


_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0))


def _edge_coords(edge, t):
    return ((-1 + 2 * t, -1.), (1., -1 + 2 * t), (1 - 2 * t, 1.), (-1., 1 - 2 * t))[edge]


def _face_candidates(x, c):
    """Interior stationary point and the four straight edges of one bilinear face."""
    out = []
    mixed = 0.25 * (c[0] - c[1] + c[2] - c[3])
    # Admissibility (proof of a unique projection on this face): the tangents are
    # a = A + eta*m, b = B + xi*m, so sigma_min([a b]) >= sigma_min([A B]) - sqrt(2)|m|
    # on the whole square, and f = |p - x|^2 / 2 has Hessian J^T J + [[0, r.m], [r.m, 0]]
    # with |r.m| <= dmax |m| (dmax: farthest corner, a convex maximum). If the lower
    # bound below is positive, f is strictly convex on the face, so the stationary
    # point (interior Newton) or the boundary minimiser (the four straight edges) is
    # the unique closest point. Otherwise the face is refused, never approximated.
    big_a = 0.25 * (-c[0] + c[1] + c[2] - c[3])
    big_b = 0.25 * (-c[0] - c[1] + c[2] + c[3])
    twist = float(np.linalg.norm(mixed))
    sigma = float(np.linalg.svd(np.column_stack((big_a, big_b)), compute_uv=False)[-1]
                  ) - np.sqrt(2.0) * twist
    dmax = float(np.max(np.linalg.norm(c - x, axis=1)))
    if not np.all(np.isfinite(c)) or sigma <= 0. or sigma * sigma - dmax * twist <= 0.:
        raise CornerContactRefusal(
            "inadmissible_face", "bilinear face is too warped for a proven unique projection")
    xi = eta = 0.
    for _ in range(60):
        p = _shape(xi, eta) @ c
        a, b = _tangents(c, xi, eta)
        r = p - x
        f = np.array((r @ a, r @ b))
        jac = np.array(((a @ a, b @ a + r @ mixed), (a @ b + r @ mixed, b @ b)))
        try:
            step = np.linalg.solve(jac, -f)
        except np.linalg.LinAlgError as error:
            raise CornerContactRefusal(
                "unresolved_interior_projection", "singular interior projection system") from error
        xi, eta = xi + step[0], eta + step[1]
        if not (np.isfinite(xi) and np.isfinite(eta)):
            raise CornerContactRefusal(
                "unresolved_interior_projection", "interior projection diverged")
        if float(np.max(np.abs(step))) < 1e-15:
            if abs(xi) <= 1. and abs(eta) <= 1.:
                p = _shape(xi, eta) @ c
                out.append((float(np.linalg.norm(x - p)), float(xi), float(eta)))
            break
    else:
        raise CornerContactRefusal(
            "unresolved_interior_projection", "interior projection did not converge")
    for edge, (i, j) in enumerate(_EDGES):
        d = c[j] - c[i]
        t = min(1., max(0., float((x - c[i]) @ d) / float(d @ d)))
        e_xi, e_eta = _edge_coords(edge, t)
        p = _shape(e_xi, e_eta) @ c
        out.append((float(np.linalg.norm(x - p)), e_xi, e_eta))
    return out


def _face_minimum_bounds(x, corners, candidates):
    """Exact-rational bounds for the minimum of |Q4(z)-x|^2/2 on [-1,1]^2.

    This refines a coarse distance tie, never replaces the bilinear projection.
    A Gershgorin bound m>0 proves strong convexity of the squared distance on
    the entire parameter square. For any feasible candidate z, minimizing
    f(z)+grad(f)(z).d+m*|d|^2/2 over that square gives a rigorous lower bound;
    f(z) itself is an upper bound. Fractions avoid comparison roundoff.
    """
    from fractions import Fraction as F
    c = [[F(float(v)) for v in row] for row in corners]
    xx = [F(float(v)) for v in x]
    dot = lambda a, b: sum(ai * bi for ai, bi in zip(a, b))
    center = [sum(row[i] for row in c) / 4 for i in range(3)]
    a = [(-c[0][i]+c[1][i]+c[2][i]-c[3][i])/4 for i in range(3)]
    b = [(-c[0][i]-c[1][i]+c[2][i]+c[3][i])/4 for i in range(3)]
    mix = [(c[0][i]-c[1][i]+c[2][i]-c[3][i])/4 for i in range(3)]
    am, bm, mm = dot(a, mix), dot(b, mix), dot(mix, mix)
    off = abs(dot(a, b)+dot([p-q for p,q in zip(center,xx)],mix))
    off += 2*abs(am)+2*abs(bm)+2*mm
    m = min(dot(a,a)-2*abs(am), dot(b,b)-2*abs(bm))-off
    if m <= 0:
        return None  # No new admission when this sufficient proof is unavailable.
    best = None
    for row in candidates:
        xi, eta = F(float(row[2])), F(float(row[3]))
        residual = [center[i]+a[i]*xi+b[i]*eta+mix[i]*xi*eta-xx[i] for i in range(3)]
        upper = dot(residual,residual)/2
        grad = (dot([a[i]+eta*mix[i] for i in range(3)],residual),
                dot([b[i]+xi*mix[i] for i in range(3)],residual))
        d = [max(F(-1),min(F(1),z-g/m))-z for z,g in zip((xi,eta),grad)]
        lower = upper+sum(g*di+m*di*di/2 for g,di in zip(grad,d))
        if best is None or upper < best[0]:
            best = upper, lower, row
    return best


def _refine_distance_tie(x, positions, slices, found, band):
    """Discard only candidates proved not to minimize their face/face union."""
    bounded = []
    for face in sorted({r[1] for r in band}):
        start, end = slices[face]
        bounds = _face_minimum_bounds(x,positions[start:end],
                                      [r for r in found if r[1] == face])
        if bounds is None:
            return None
        bounded.append(bounds)
    best_upper = min(r[0] for r in bounded)
    possible = [r[2] for r in bounded if r[1] <= best_upper]
    # An exact-input proof must also distinguish the rounded distance values.
    # Equal/unresolved floating distances remain refused, including the captured
    # physical44/45 ambiguity; tiny accidental input asymmetry is not a tie-break.
    if len(possible) == 1 and len(bounded) > 1:
        winner = possible[0]
        if any(np.nextafter(winner[0],np.inf) >= r[2][0]
               for r in bounded if r[2][1] != winner[1]):
            return None
    return possible


def closest_panel_point(x, panel_positions, face_slices, face_nodes, panel_director_gradient,
                        faces=None):
    """Closest point on the union of ``faces``; returns a dict or refuses."""
    faces = range(len(face_slices)) if faces is None else faces
    found = []
    for j in faces:
        a, b = face_slices[j]
        for dist, xi, eta in _face_candidates(x, panel_positions[a:b]):
            found.append((dist, j, xi, eta))
    if not found:
        raise CornerContactRefusal("no_candidate_face", "no panel face to project on")
    best = min(f[0] for f in found)
    band = [row for row in found if row[0]-best <= 1e-9]
    first = band[0]
    start,end = face_slices[first[1]]
    p0 = _shape(first[2],first[3]) @ panel_positions[start:end]
    if any(np.linalg.norm(_shape(row[2],row[3]) @ panel_positions[
            face_slices[row[1]][0]:face_slices[row[1]][1]]-p0) > TIE_M for row in band):
        refined = _refine_distance_tie(x,panel_positions,face_slices,found,band)
        if refined is not None:
            found = refined
            best = min(row[0] for row in found)
    point = None
    owners = []
    for dist, j, xi, eta in sorted(found):
        a, b = face_slices[j]
        p = _shape(xi, eta) @ panel_positions[a:b]
        if dist - best > 1e-9:
            break
        if point is None:
            point = p
        if float(np.linalg.norm(p - point)) <= TIE_M:
            if all(o[0] != j for o in owners):
                owners.append((j, xi, eta))
        else:
            raise CornerContactRefusal(
                "nonunique_closest_point",
                f"distinct closest points tie within 1e-9 m (faces {owners[0][0]}, {j})")
    owners.sort()
    if len(owners) > 1:
        reference = _node_weights(face_nodes, face_slices, owners[0])
        for owner in owners[1:]:
            other = _node_weights(face_nodes, face_slices, owner)
            if set(reference) != set(other) or any(
                    abs(reference[n] - other[n]) > 1e-12 for n in reference):
                raise CornerContactRefusal(
                    "inconsistent_seam", "owner faces disagree on shared-node weights")
        for node in reference:
            rows = []
            for j, xi, eta in owners:
                a, _ = face_slices[j]
                local = face_nodes[j].index(node)
                rows.append(a + local)
            for row in rows[1:]:
                if (np.max(np.abs(panel_positions[row] - panel_positions[rows[0]])) > TIE_M
                        or np.max(np.abs(panel_director_gradient[row]
                                         - panel_director_gradient[rows[0]])) > TIE_M):
                    raise CornerContactRefusal(
                        "inconsistent_seam", "shared panel corner copies differ")
    d = np.asarray(x) - point
    dist = float(np.linalg.norm(d))
    normals = []
    for j, xi, eta in owners:
        a, b = face_slices[j]
        t1, t2 = _tangents(panel_positions[a:b], xi, eta)
        n = np.cross(t1, t2)
        length = float(np.linalg.norm(n))
        if not np.isfinite(length) or length <= 0.:
            raise CornerContactRefusal("degenerate_face", "degenerate panel face normal")
        normals.append(n / length)
    if dist < ZERO_DISTANCE_M:
        if len(owners) > 1:
            raise CornerContactRefusal(
                "seam_zero_distance", "corner on a panel seam at zero distance")
        xi0, eta0 = owners[0][1], owners[0][2]
        if abs(xi0) > 1. - 1e-12 or abs(eta0) > 1. - 1e-12:
            raise CornerContactRefusal(
                "zero_distance_boundary", "corner on a panel face boundary at zero distance")
        u = normals[0]
        g = float(u @ d)
    else:
        signs = {int(np.sign(d @ n)) for n in normals}
        if len(signs) != 1 or 0 in signs:
            raise CornerContactRefusal(
                "unresolved_side", "closest point has no unique side of the panel")
        s = float(signs.pop())
        u = s * d / dist
        g = s * dist
    j, xi, eta = owners[0]
    return dict(gap=g, direction=u, face=j, xi=xi, eta=eta, weights=_shape(xi, eta),
                owners=[o[0] for o in owners], distance=dist)


def _node_weights(face_nodes, face_slices, owner):
    j, xi, eta = owner
    w = _shape(xi, eta)
    return {n: float(w[k]) for k, n in enumerate(face_nodes[j]) if abs(w[k]) > 1e-14}


def corner_gaps(sphere_positions, panel_map, panel_pose, *, radius=0.05,
                multipliers=None, query_position_errors=None, return_audit=False):
    """Gaps for every sphere face-local corner within ``radius`` of the panel.

    Pruning proof: every original bilinear face lies in the axis box of its four
    corners; a corner farther than ``radius`` (infinity norm) from all boxes is
    farther than ``radius`` from the surface, so g >= radius and it cannot be
    active. A corner horizontally over the panel but deeper than ``radius`` below
    a face box refuses (it cannot be pruned as "above").
    """
    pos = panel_pose.positions
    slices = panel_map.face_slices
    nodes = [f.node_ids for f in panel_map.faces]
    lo = np.array([pos[a:b].min(axis=0) for a, b in slices])
    hi = np.array([pos[a:b].max(axis=0) for a, b in slices])
    result, inactive_audit = {}, []
    for i, x in enumerate(np.asarray(sphere_positions, dtype=float)):
        if not np.all(np.isfinite(x)):
            raise CornerContactRefusal("nonfinite_corner", "nonfinite sphere corner")
        horizontal = np.all((x[:2] >= lo[:, :2]) & (x[:2] <= hi[:, :2]), axis=1)
        if np.any(horizontal & (x[2] < lo[:, 2] - radius)):
            raise CornerContactRefusal(
                "corner_beyond_panel", f"sphere corner {i} is far below the panel skin")
        near = np.nonzero(np.all((x >= lo - radius) & (x <= hi + radius), axis=1))[0]
        if near.size == 0:
            continue
        try:
            info = closest_panel_point(x, pos, slices, nodes, panel_pose.director_gradient,
                                       faces=[int(j) for j in near])
        except CornerContactRefusal as refusal:
            # Never manufacture a closest owner/gradient for an ambiguous point.
            # Explicit current zero force + a repeated positive gap certificate
            # eliminates exactly the inactive complementarity branch.
            proof = None
            if (refusal.code == "nonunique_closest_point" and return_audit
                    and multipliers is not None and multipliers.get(i, 0.) == 0.
                    and query_position_errors is not None
                    and getattr(panel_pose, "position_errors", None) is not None):
                from .corner_contact_projection_inactive import separated_zero_force_proof
                proof = separated_zero_force_proof(
                    x, query_position_errors[i], pos, panel_pose.position_errors,
                    slices, near, radius=radius, zero_distance=ZERO_DISTANCE_M)
            if proof is None:
                raise
            proof["corner_id"] = int(i)
            inactive_audit.append(proof)
            continue
        if info["distance"] <= radius:
            result[i] = info
    return (result, inactive_audit) if return_audit else result
