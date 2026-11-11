"""Read-only elastic S3 covector components, using the owner's actual chart."""
import numpy as np
from ._ge_beam3_variational_shell import deformation
from .e4_pl_s3_v2d_element import FORMULATION_ID


def conservative_components(model, element, displacement, origin, layers, *,
                            rotation_anchors=None, stable_translation=False):
    if type(stable_translation) is not bool:
        raise ValueError('actual owned S3 translation arithmetic policy required')
    material = model.get_material(element.material_name)
    if (getattr(element, 'formulation_id', None) != FORMULATION_ID or element.num_nodes != 3
            or element.shell_section is not None or material.yield_stress != 0.
            or material.hardening_curve is not None or origin is None):
        raise ValueError('conservative S3 capture requires owned elastic V2D section')
    reference = np.array([model.mesh.nodes[node].coords() for node in element.node_ids])
    local, differential, second = deformation(reference, displacement,
        rotation_anchors=rotation_anchors, stable_translation=stable_translation)
    force, tangent, candidate = element.compute_nonlinear_response(model.mesh, material,
                                                                  local, origin, layers, True)
    if np.linalg.norm(tangent-tangent.T) > 1e-11*max(1., np.linalg.norm(tangent)):
        raise ValueError('nonconservative local S3 elastic tangent')
    km = differential.T@tangent@differential
    geometric = np.einsum('i,ijk->jk', force, second)
    covector = differential.T@force
    if not all(np.isfinite(a).all() for a in (km, geometric, covector)):
        raise ValueError('finite conservative elastic S3 components required')
    return covector, km+geometric, km, geometric, candidate
