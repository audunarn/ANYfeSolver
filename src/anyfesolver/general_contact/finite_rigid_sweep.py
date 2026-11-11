"""Whole original additive-coordinate path; fixed target, enclosed Exp skins."""
import numpy as np
from .finite_rotation_skin import _up, curvature_radius, uniform_skin_evaluation_error
from .finite_rotation_sweep import _pose
from .finite_rigid_support import tool_normal_gap_lower


def finite_rigid_path_admissible(manager, start, end):
    start, end = np.asarray(start, float), np.asarray(end, float)
    skin = manager.finite_skin.source
    if start.shape != (skin.n_dof,) or end.shape != start.shape or not np.isfinite(start).all() or not np.isfinite(end).all():
        raise ValueError('finite rigid path requires complete finite coordinates')
    manager.require_finite_deformation(start)
    try:
        manager.require_finite_deformation(end)
        manager._positions(end)
    except ValueError:
        return False
    target = manager.reference[1]
    et = np.zeros_like(target)

    def lower(a, b, ea, eb):
        return min(tool_normal_gap_lower(a[si], b[si], target[ti],
            manager.law.source_faces[i].reference_positions,
            manager.law.target_faces[j].reference_positions,
            endpoint_errors=(ea[si], eb[si]))
            for i, si in enumerate(manager.law.source_face_indices)
            for j, ti in enumerate(manager.law.target_face_indices))

    left, right = _pose(skin, start, end, 0.), _pose(skin, start, end, 1.)
    if lower(left[2].positions, left[2].positions, left[3], left[3]) < -manager.gap_tolerance:
        raise ValueError('committed finite rigid chart/gap is uncertified')
    if lower(right[2].positions, right[2].positions, right[3], right[3]) < -manager.gap_tolerance:
        return False

    def certify(a, b, pa, pb, depth):
        curve = curvature_radius(pa[0][skin.dofs[:, 3:]], pb[0][skin.dofs[:, 3:]],
            skin.offsets, skin.directors,
            coordinate_errors=(pa[1][skin.dofs[:, 3:]], pb[1][skin.dofs[:, 3:]]))
        evaluation = uniform_skin_evaluation_error(skin, pa[0], pb[0], pa[1], pb[1])
        radius = _up(curve[:, None] + evaluation)
        ea, eb = _up(pa[3]+radius), _up(pb[3]+radius)
        from .finite_rigid_support import slab_support_and_ownership
        if (lower(pa[2].positions, pb[2].positions, ea, eb) >= -manager.gap_tolerance
                and slab_support_and_ownership(manager, pa[2].positions, pb[2].positions, ea, eb)):
            return True
        if depth >= manager.settings.maximum_swept_subdivisions:
            return False
        middle = (a+b)/2.
        pm = _pose(skin, start, end, middle)
        return certify(a, middle, pa, pm, depth+1) and certify(middle, b, pm, pb, depth+1)

    return certify(0., 1., left, right, 0)
