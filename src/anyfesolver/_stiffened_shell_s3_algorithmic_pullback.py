"""Experimental objective S3 J2 algorithmic pullback from a fixed predecessor.

The local S3 material update is unchanged. Its algorithmic derivative is pulled
through the full deformational map and converted to spatial residual rows and
additive shell-chart columns. This does not assert a plastic energy potential.
"""

import numpy as np

from ._stiffened_shell_algorithmic_pullback import spatial_response
from ._ge_beam3_variational_shell import deformation
from .e4_pl_s3_v2d_element import FORMULATION_ID
from .material_curves import DNVC208MaterialCurve


POLICY = "STIFFENED_SHELL_S3_J2_ALGORITHMIC_PULLBACK_V1"


def local_coordinates(model, element, displacement, *, rotation_anchors=None):
    reference = np.array([model.mesh.nodes[node].coords()
                          for node in element.node_ids])
    return deformation(reference, displacement, stable_translation=True,
                       rotation_anchors=rotation_anchors)[0]


def response(model, element, displacement, origin, layers, *, rotation_anchors=None,
             capture_material_trial=False):
    """Return one objective trial and owned local material candidate.

    With d(q), D=dd/dq: g=D.T f and dg/dq=D.T K_alg D+sum(f_i Hess d_i).
    The chart correction differentiates spatial rows, not a symmetrized K_alg.
    Every call uses the caller's same accepted local constitutive predecessor.
    """
    material = model.get_material(element.material_name)
    if type(capture_material_trial) is not bool:
        raise ValueError("explicit S3 material trial capture policy required")
    if (getattr(element, "formulation_id", None) != FORMULATION_ID or
            element.num_nodes != 3 or element.shell_section is not None or
            not isinstance(material.hardening_curve, DNVC208MaterialCurve) or
            origin is None):
        raise ValueError("experimental S3 isotropic J2 accepted predecessor required")
    reference = np.array([model.mesh.nodes[node].coords()
                          for node in element.node_ids])
    u = np.asarray(displacement, dtype=float)
    local, differential, second = deformation(reference, u, stable_translation=True,
                                              rotation_anchors=rotation_anchors)
    force, tangent, candidate = element.compute_nonlinear_response(
        model.mesh, material, local, origin, layers, True)
    force, tangent = np.asarray(force), np.asarray(tangent)
    if (force.shape != (18,) or tangent.shape != (18, 18) or
            not np.isfinite(force).all() or not np.isfinite(tangent).all() or
            candidate is None):
        raise ValueError("finite S3 J2 algorithmic trial required")
    residual, spatial_tangent = spatial_response(u, force, tangent, differential, second)
    if capture_material_trial:
        from ._stiffened_shell_s3_material_event import capture
        return residual, spatial_tangent, candidate, capture(
            model, element, u, differential, origin, candidate, layers)
    return residual, spatial_tangent, candidate
