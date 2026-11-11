"""Explicit Q4 elastic/J2 full-map adapter; ordinary Q4 routes are unchanged."""

import numpy as np

from ._ge_beam3_variational_shell import deformation
from ._stiffened_shell_algorithmic_pullback import spatial_response
from .e4_pl_element import FORMULATION_ID
from .material_curves import DNVC208MaterialCurve


POLICY = "STIFFENED_SHELL_Q4_ALGORITHMIC_PULLBACK_V1"
HISTORICAL_POLICY = "historical_corotational"


def local_coordinates(model, element, displacement, *, rotation_anchors=None):
    reference = np.array([model.mesh.nodes[node].coords() for node in element.node_ids])
    return deformation(reference, displacement, stable_translation=True,
                       rotation_anchors=rotation_anchors)[0]


def response(model, element, displacement, origin, layers, *, rotation_anchors=None,
             prepared_correction=None, stable_rotation=False, capture_material_trial=False):
    """Differentiate the unchanged local update from one accepted predecessor."""
    if type(capture_material_trial) is not bool:
        raise ValueError("explicit Q4 material trial capture policy required")
    material = model.get_material(element.material_name)
    curve = material.hardening_curve
    if (getattr(element, "formulation_id", None) != FORMULATION_ID or
            element.num_nodes != 4 or element.shell_section is not None or
            (curve is not None and not isinstance(curve, DNVC208MaterialCurve)) or
            (curve is None and material.yield_stress != 0.) or origin is None):
        raise ValueError("experimental Q4 elastic/isotropic J2 accepted predecessor required")
    reference = np.array([model.mesh.nodes[node].coords() for node in element.node_ids])
    u = np.asarray(displacement, dtype=float)
    local, differential, second = deformation(reference, u, stable_translation=True,
                                              rotation_anchors=rotation_anchors,
                                              stable_rotation=stable_rotation)
    if prepared_correction is None:
        force, tangent, candidate = element.compute_nonlinear_response(
            model.mesh, material, local, origin, layers, True)
    else:
        force, tangent, candidate = element._compute_prepared_nonlinear_response(
            model.mesh, material, local, origin, layers, True,
            prepared=prepared_correction)
    force, tangent = np.asarray(force), np.asarray(tangent)
    if (force.shape != (24,) or tangent.shape != (24, 24) or
            not np.isfinite(force).all() or not np.isfinite(tangent).all() or
            candidate is None):
        raise ValueError("finite Q4 algorithmic trial required")
    residual, spatial_tangent = spatial_response(u, force, tangent, differential, second,
                                                 stable_rotation=stable_rotation)
    if capture_material_trial:
        from ._stiffened_shell_q4_material_event import capture
        payload = capture(model, element, u, local, differential, origin, candidate, layers)
        return residual, spatial_tangent, candidate, payload
    return residual, spatial_tangent, candidate


def conservative_components(model, element, displacement, origin, layers, *,
                            rotation_anchors=None, prepared_correction=None):
    """Read-only elastic covector Hessian in the actual additive shell chart.

    Unlike a spatial-row tangent, these blocks are derivatives of one scalar
    potential. The unchanged qualified linear correction contributes to both
    material stiffness and the force contracted with deformation curvature.
    """
    material = model.get_material(element.material_name)
    if (getattr(element, 'formulation_id', None) != FORMULATION_ID or
            element.num_nodes != 4 or element.shell_section is not None or
            material.hardening_curve is not None or material.yield_stress != 0. or origin is None):
        raise ValueError('conservative Q4 capture requires the owned elastic section')
    reference = np.array([model.mesh.nodes[node].coords() for node in element.node_ids])
    local, differential, second = deformation(reference, displacement,
        stable_translation=True, rotation_anchors=rotation_anchors)
    if prepared_correction is None:
        force, tangent, candidate, parts = element.compute_nonlinear_response(
            model.mesh, material, local, origin, layers, True, _return_tangent_components=True)
    else:
        force, tangent, candidate, parts = element._compute_prepared_nonlinear_response(
            model.mesh, material, local, origin, layers, True,
            prepared=prepared_correction, _return_tangent_components=True)
    km = differential.T @ parts['material'] @ differential
    geometric = differential.T @ parts['geometric'] @ differential + np.einsum('i,ijk->jk', force, second)
    hessian = differential.T @ tangent @ differential + np.einsum('i,ijk->jk', force, second)
    covector = differential.T @ force
    if not all(np.isfinite(a).all() for a in (km, geometric, hessian, covector)):
        raise ValueError('finite conservative Q4 components required')
    return covector, hessian, km, geometric, candidate
