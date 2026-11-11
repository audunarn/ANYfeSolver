"""Per-patch native/S3 certificate along the retained static solver path.

Shared with the original single-patch owner; the native element index selects
its own physical cell rotations inside the 24-coordinate retained block.
"""
import numpy as np


def native_s3_swept_admissible(context, work, beam_index, gap_tolerance,
        maximum_intervals, recoverable_error, start, u_start, increment, end,
        u_end, start_gap, end_gap, start_rim, end_rim, *,
        geometry_only=False, projected_gap=False, geometry_query=None,
        material_radius=None):
    if material_radius is not None and (not np.isfinite(material_radius) or material_radius<=0.):
        raise ValueError('positive finite material radius required for surface path')
    arm=0. if material_radius is None else float(material_radius)
    if min(start_gap, end_gap) < -gap_tolerance:
        return False
    first = work.shell.current_positions(u_start)
    last = work.shell.current_positions(u_end)
    a0, b0 = first[1]-first[0], first[2]-first[0]
    a1, b1 = last[1]-last[0], last[2]-last[0]
    c0, c2 = np.cross(a0, b0), np.cross(a1, b1)
    middle = np.cross(0.5*(a0+a1), 0.5*(b0+b1))
    c1 = 2.0*middle-0.5*(c0+c2)
    normal0 = c0/np.linalg.norm(c0)
    area_lower = min(float(normal0@item) for item in (c0, c1, c2))
    if area_lower <= work.contact.geometry_tolerance**2:
        return False
    if material_radius is not None:
        # physical_pose requires orientation against the REFERENCE facet, not
        # merely the start normal. Its projection is quadratic along this
        # affine shell path; positive Bernstein controls certify the whole path.
        reference=work.shell.reference_positions
        reference_cross=np.cross(reference[1]-reference[0],reference[2]-reference[0])
        reference_area=float(np.linalg.norm(reference_cross))
        if not np.isfinite(reference_area) or reference_area<=work.contact.geometry_tolerance**2:
            return False
        reference_normal=reference_cross/reference_area
        orientation_lower=min(float(reference_normal@c) for c in (c0,c1,c2))
        arithmetic=256.*np.finfo(float).eps*max(reference_area,
            *(float(np.linalg.norm(c)) for c in (c0,c1,c2)))
        if orientation_lower<=max(work.contact.geometry_tolerance**2,arithmetic):
            return False
    cross_speed = (
        np.linalg.norm(a1-a0)*max(np.linalg.norm(b0), np.linalg.norm(b1))
        +np.linalg.norm(b1-b0)*max(np.linalg.norm(a0), np.linalg.norm(a1))
    )
    normal_speed = cross_speed/area_lower
    nodes = context.nodes[beam_index]
    before = start.positions[nodes]+start.position_low[nodes]
    after = end.positions[nodes]+end.position_low[nodes]
    lift = max(np.linalg.norm(work.contact.beam_reference.half_cell_lift(c, 0.5))
               for c in (0, 1))
    angular = max(np.linalg.norm(increment[context.nodal_count+24*beam_index+3*c:
                                           context.nodal_count+24*beam_index+3*c+3])
                  for c in (0, 1))
    node_speed = float(np.max(np.linalg.norm(after-before, axis=1)))
    beam_speed = node_speed+angular*(lift+arm)
    bezier_speed = node_speed+2.0*angular*lift+angular*arm
    shell_speed = float(np.linalg.norm(last[0]-first[0]))
    radius = max(float(np.linalg.norm(point-base))
                 for points, base in ((before, first[0]), (after, last[0]))
                 for point in points)+lift+arm
    lipschitz = beam_speed+shell_speed+normal_speed*radius
    if projected_gap:
        # g=n.(r-centroid)-R is identical to the vertex-anchor gap. The
        # centroid reduces the radius multiplying dn/dt. For unit normals,
        # |n(t)-n(0)| <= min(2, sup|dn/dt|) on this unit path interval.
        center0=np.mean(first,axis=0); center1=np.mean(last,axis=0)
        excursion=min(2.,float(normal_speed))
        relative_velocity=(after-before)-(center1-center0)
        translation=max(abs(float(normal0@v))+excursion*float(np.linalg.norm(v))
                        for v in relative_velocity)
        cell_angular=[increment[context.nodal_count+24*beam_index+3*c:
                               context.nodal_count+24*beam_index+3*c+3] for c in (0,1)]
        rotation=max(float(np.linalg.norm(np.cross(normal0,v)))
                     +excursion*float(np.linalg.norm(v)) for v in cell_angular)*(lift+arm)
        centered_radius=max(float(np.linalg.norm(point-base))
            for points,base in ((before,center0),(after,center1)) for point in points)+lift+arm
        lipschitz=translation+rotation+normal_speed*centered_radius
    rim_speed = 0.0
    for index in range(3):
        j = (index+1) % 3
        edge_first = first[j]-first[index]
        edge_last = last[j]-last[index]
        edge_change = edge_last-edge_first
        denominator = float(edge_change@edge_change)
        station = (0.0 if denominator == 0.0 else float(np.clip(
            -float(edge_first@edge_change)/denominator, 0.0, 1.0,
        )))
        edge_lower = float(np.linalg.norm(edge_first+station*edge_change))
        if edge_lower <= work.contact.geometry_tolerance:
            return False
        inward_speed = normal_speed+float(np.linalg.norm(edge_change))/edge_lower
        vertex_speed = float(np.linalg.norm(last[index]-first[index]))
        vertex_radius = max(float(np.linalg.norm(point-base))
                            for points, base in ((before, first[index]),
                                                 (after, last[index]))
                            for point in points)+2.0*lift+arm
        rim_speed = max(rim_speed,
                        bezier_speed+vertex_speed+inward_speed*vertex_radius)
    roundoff = 256.0*np.finfo(float).eps*max(
        1.0, float(np.max(np.abs(before))), float(np.max(np.abs(after))),
        float(np.max(np.abs(first))), float(np.max(np.abs(last))),
    )
    rim_required = (work.contact.rim_clearance if material_radius is not None else
                    work.contact.radius+work.contact.rim_clearance)
    if min(start_rim, end_rim)-roundoff < rim_required:
        return False
    stack = [(0.0, 1.0, start_gap, end_gap,
              start_rim-rim_required, end_rim-rim_required)]
    visited = 0
    while stack:
        left, right, lower_left, lower_right, rim_left, rim_right = stack.pop()
        visited += 1
        if visited > maximum_intervals:
            return False
        gap_certified = (min(lower_left, lower_right)
                         -0.5*(right-left)*lipschitz-roundoff
                         >= -gap_tolerance)
        rim_certified = (min(rim_left, rim_right)
                         -0.5*(right-left)*rim_speed-roundoff >= 0.0)
        if gap_certified and rim_certified:
            continue
        t = 0.5*(left+right)
        try:
            state = context.advance(start, t*increment[:context.count])
            u = u_start+t*(u_end-u_start)
            if geometry_only:
                if geometry_query is not None:
                    gap,rim=geometry_query(state,u)
                else:
                    gap,rim=work.contact.geometry_gap(state.positions[nodes],state.position_low[nodes],
                        state.cell_rotations[beam_index],work.shell.current_positions(u))
            else:
                middle_contact = work.evaluate(state.positions[nodes], state.position_low[nodes],
                        state.cell_rotations[beam_index], u).contact
                gap=middle_contact.minimum_gap_m; rim=middle_contact.minimum_rim_clearance_m
        except (ValueError, MemoryError) as exc:
            if recoverable_error(exc):
                return False
            raise
        rim -= rim_required
        if gap < -gap_tolerance:
            return False
        stack.extend(((left, t, lower_left, gap, rim_left, rim),
                      (t, right, gap, lower_right, rim, rim_right)))
    return True
