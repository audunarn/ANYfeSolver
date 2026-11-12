"""Call-local quadrature frames for the exact legacy eight-node shell.

The scalar element remains the oracle, including all degenerate-map errors.
No mesh, material, displacement or element data survives the invocation.
"""

import numpy as np


def native_recovery_frames(element, coords, points, material):
    from . import elements
    from .elements import (
        Element, ShellElement, _SMALL, _SHELL_RECOVERY_FRAME_METHODS,
        _SHELL_RECOVERY_BASE_METHODS,
        _RecoveryIsotropicMaterial, _SHELL_RECOVERY_MATERIAL_METHODS,
        _SHELL_RECOVERY_FRAME_HELPERS,
    )

    if type(element) is not ShellElement:
        return None
    instance = vars(element)
    for cls, expected in ((ShellElement, _SHELL_RECOVERY_FRAME_METHODS),
                          (Element, _SHELL_RECOVERY_BASE_METHODS)):
        namespace = vars(cls)
        if len(namespace) != len(expected) or any(
            namespace.get(name) is not method or name in instance
            for name, method in expected
        ):
            return None
    if type(element.node_ids) is not list or len(element.node_ids) != 8:
        return None
    if any(getattr(elements, name) is not method for name, method in _SHELL_RECOVERY_FRAME_HELPERS):
        return None
    if type(material) is not _RecoveryIsotropicMaterial:
        return None
    namespace = vars(_RecoveryIsotropicMaterial)
    if len(namespace) != len(_SHELL_RECOVERY_MATERIAL_METHODS) or any(
        namespace.get(name) is not method for name, method in _SHELL_RECOVERY_MATERIAL_METHODS
    ):
        return None
    values = vars(material)
    if set(values) != {"name", "elastic_modulus", "poisson_ratio", "density", "yield_stress", "hardening_curve"}:
        return None
    if type(values["name"]) is not str or values["hardening_curve"] is not None:
        return None
    if any(type(values[name]) not in (float, int) for name in ("elastic_modulus", "poisson_ratio", "density", "yield_stress")):
        return None
    if (type(element.thickness) is not float or type(element.material_angle_deg) is not float
            or element.material_direction is not None):
        return None
    shapes = [element.compute_shape_functions(float(x), float(y)) for x, y in points]
    # Keep the original small matrix-vector products for the geometry map.
    jacobians = np.asarray([element.compute_jacobian(coords, a, b) for _, a, b in shapes])
    tx, ty = jacobians[:, 0], jacobians[:, 1]

    def dot(a, b):
        # Preserve the scalar BLAS reduction order: small frame roundoff can
        # amplify in bending recovery for thin shells.
        return (a[:, None, :] @ b[:, :, None])[:, 0, 0]

    def cross(a, b):
        # These inputs always contain three Cartesian components.  Avoid the
        # general axis/broadcast dispatch in np.cross while retaining _cross3's
        # component operation order.
        return np.column_stack((
            a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
            a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
            a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0],
        ))

    def normalize(a):
        lengths = np.sqrt(dot(a, a))
        # Near the scalar rejection boundary use the scalar oracle itself;
        # this admission margin is not a changed physical/map tolerance.
        if np.any(~np.isfinite(lengths)) or np.any(lengths < 10.0 * _SMALL):
            return None
        return a / lengths[:, None]

    e3 = normalize(cross(tx, ty))
    if e3 is None:
        return None
    e1 = normalize(tx - dot(tx, e3)[:, None] * e3)
    if e1 is None:
        return None
    e2 = normalize(cross(e3, e1))
    if e2 is None:
        return None
    e1 = normalize(cross(e2, e3))
    if e1 is None:
        return None
    a, b, c, d = dot(tx, e1), dot(tx, e2), dot(ty, e1), dot(ty, e2)
    determinant = a * d - b * c
    if np.any(~np.isfinite(determinant)) or np.any(np.abs(determinant) < 10.0 * _SMALL):
        return None
    inv00, inv01 = d / determinant, -b / determinant
    inv10, inv11 = -c / determinant, a / determinant
    dxi = np.asarray([shape[1] for shape in shapes])
    deta = np.asarray([shape[2] for shape in shapes])
    dx = inv00[:, None] * dxi + inv01[:, None] * deta
    dy = inv10[:, None] * dxi + inv11[:, None] * deta
    frames = np.stack((e1, e2, e3), axis=2)
    return tuple((shape[0], frames[i], dx[i], dy[i]) for i, shape in enumerate(shapes))
