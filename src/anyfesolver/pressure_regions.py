"""Consistent equivalent nodal loads of a pressure acting on a PARTIAL (wet) reference region of a shell element.

``PressurePatch``/``LoadCase.add_pressure`` select whole elements.  This adapter integrates the element's OWN
pressure virtual work  ``f_i = sign * integral_{wet region} p g_i dA``  over the sub-region where a signed linear
level field is positive.  Two interpolation sources, no mechanics duplicated or modified:

* native density (current admitted S3 ``e4-pl-s3-v2d`` and its V2C base): the element's
  ``compute_native_pressure_density`` is the single owner of the translational AND rotational (l, m) terms that its
  uniform dead load also uses; the full 6-DOF nodal vector is carried end to end;
* shape functions (legacy/qualified Q4, higher-order S6/S8): ``N_i`` with the element Jacobian and
  ``sheet_area_orientation_sign``; the actual pressure interpolation has no rotational terms, so moments are zero.

Whenever an element also has a native uniform dead-pressure load, the all-18/24-component equivalence for uniform
pressure is REQUIRED (``element_kernel``); otherwise the formulation is refused (no silent substitution, no dropped
rotational work).  Corner level/pressure fields are linear over the reference triangle (S3/S6) or piecewise linear
over four reference sub-triangles about the centre (quads; centre = corner mean).  Clipping is delegated to the
ANYgeometry public contract, imported lazily.
"""
from dataclasses import dataclass
import math
import numpy as np

SUPPORTED_NODE_COUNTS = (3, 4, 6, 8)
_SMALL = 1e-14


class PressureRegionError(ValueError):
    """Unsupported formulation, degenerate geometry or non-finite field."""


@dataclass(frozen=True)
class RegionLoad:
    node_ids: tuple
    forces: np.ndarray        # (n_nodes, 3) equivalent nodal forces
    wet_area: float           # physical area of the wet region
    total_area: float         # physical area of the full element (same quadrature)
    wet_fraction: float       # wet_area / total_area
    mean_pressure: float      # full-area element-average pressure (integral of p over wet region / total area)
    state: str                # 'dry' | 'partial' | 'wet'
    moments: np.ndarray = None  # (n_nodes, 3) direct nodal moments (zero where the interpolation has none)

    def __post_init__(self):
        if self.moments is None:
            object.__setattr__(self, 'moments', np.zeros_like(self.forces))

    @property
    def force(self):
        return self.forces.sum(axis=0)

    @property
    def vectors(self):
        """(n_nodes, 6) consistent nodal vectors [Fx, Fy, Fz, Mx, My, Mz]."""
        return np.hstack([self.forces, self.moments])


def _reference(n_corner):
    if n_corner == 3:
        corners = np.array([[0., 0.], [1., 0.], [0., 1.]])
        return corners, [(0, 1, 2)]
    corners = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.], [0., 0.]])
    # four triangles about the centre (index 4); centre value is the corner mean
    return corners, [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)]


def _det2(a, b):
    """2-D cross product (scalar determinant); np.cross on 2-vectors is deprecated."""
    return float(a[0] * b[1] - a[1] * b[0])


def _default_clipper():
    try:
        from anygeometry.wave_surface_clipping import clip_triangle_positive
    except ImportError as error:  # pragma: no cover - environment boundary
        raise PressureRegionError('ANYgeometry wave_surface_clipping is required for partial-region loads') from error
    return clip_triangle_positive


def _is_native(element):
    return callable(getattr(element, 'compute_native_pressure_density', None))


def _check_element(element):
    if not hasattr(element, 'node_ids'):
        raise PressureRegionError('element formulation has no pressure kernel')
    n = len(element.node_ids)
    if _is_native(element):
        if n != 3:
            raise PressureRegionError('native pressure density is only admitted for 3-node strict-flat shells')
        return n
    if not hasattr(element, 'compute_shape_functions') or not hasattr(element, 'get_node_coordinates') \
            or not hasattr(type(element), 'TRI_GAUSS_POINTS_7'):
        raise PressureRegionError('element formulation has no shape-function based consistent pressure path')
    if n not in SUPPORTED_NODE_COUNTS:
        raise PressureRegionError(f'unsupported shell node count {n}')
    return n


class _ShapeIntegrand:
    """Shape functions and surface Jacobians of the element's own mapping (no rotational terms)."""

    def __init__(self, element, mesh):
        self.element = element
        self.coords = np.asarray(element.get_node_coordinates(mesh), dtype=float)
        orientation = getattr(element, 'sheet_area_orientation_sign', None)
        self.orientation = float(orientation(mesh)) if callable(orientation) else 1.0
        if self.orientation not in (-1.0, 1.0):
            raise PressureRegionError('shell sheet area orientation sign must be -1 or +1')

    def evaluate(self, points):
        shapes, dxi, deta = [], [], []
        for xi, eta in points:
            n, a, b = self.element.compute_shape_functions(float(xi), float(eta))
            shapes.append(np.asarray(n, dtype=float)); dxi.append(np.asarray(a, dtype=float))
            deta.append(np.asarray(b, dtype=float))
        shapes, dxi, deta = np.array(shapes), np.array(dxi), np.array(deta)
        area_vectors = self.orientation * np.cross(dxi @ self.coords, deta @ self.coords)
        magnitude = np.linalg.norm(area_vectors, axis=1)
        if not np.isfinite(magnitude).all() or magnitude.min() < _SMALL:
            raise PressureRegionError('degenerate shell geometry (near-zero surface Jacobian)')
        vectors = np.zeros((len(shapes), shapes.shape[1], 6))
        vectors[:, :, :3] = np.einsum('mn,mk->mnk', shapes, area_vectors)
        return vectors, magnitude


class _NativeIntegrand:
    """The element's own native pressure virtual-work density (translation + rotational terms)."""

    def __init__(self, element, mesh):
        self.element, self.mesh = element, mesh

    def evaluate(self, points):
        density, area = self.element.compute_native_pressure_density(self.mesh, points)
        density = np.asarray(density, dtype=float)
        if not math.isfinite(area) or area < _SMALL or not np.isfinite(density).all():
            raise PressureRegionError('degenerate shell geometry (near-zero area)')
        measure = 2. * area            # physical area per unit reference-triangle measure d(xi) d(eta)
        return measure * density.reshape(len(density), 3, 6), np.full(len(density), measure)


def _integrand(element, mesh):
    return _NativeIntegrand(element, mesh) if _is_native(element) else _ShapeIntegrand(element, mesh)


def _rule(element):
    if _is_native(element):
        # Degree-5 triangle rule (7 points); the native density is polynomial of degree <= 3 per unit pressure.
        a1, b1, a2, b2 = 0.059715871789770, 0.470142064105115, 0.797426985353087, 0.101286507323456
        points = np.array([[1 / 3, 1 / 3], [b1, b1], [a1, b1], [b1, a1], [b2, b2], [a2, b2], [b2, a2]])
        weights = np.array([0.1125] + [0.066197076394253] * 3 + [0.062969590272414] * 3)
        return points, weights
    return (np.asarray(type(element).TRI_GAUSS_POINTS_7, dtype=float),
            np.asarray(type(element).TRI_GAUSS_WEIGHTS_7, dtype=float))


def _full_points(ref_points, triangles, rule):
    """Quadrature points/weights/hat-function values over the whole reference element."""
    points, weights, hats = [], [], []
    rp, rw = rule
    for tri in triangles:
        v = ref_points[list(tri)]
        area = .5 * abs(_det2(v[1] - v[0], v[2] - v[0]))
        bary = np.column_stack([1. - rp[:, 0] - rp[:, 1], rp[:, 0], rp[:, 1]])
        points.append(bary @ v); weights.append(2. * area * rw)
        h = np.zeros((len(rp), len(ref_points)))
        h[:, list(tri)] = bary
        hats.append(h)
    return np.vstack(points), np.concatenate(weights), np.vstack(hats)


def _corner_count(n_nodes):
    return 3 if n_nodes in (3, 6) else 4


def _hat_to_corner(hats, n_corner):
    """Express 5-vertex (quad) hat values through the 4 corners (centre = corner mean)."""
    if n_corner == 3:
        return hats
    return hats[:, :4] + hats[:, [4]] / 4.


def element_kernel(element, mesh):
    """Fully wet element: ``vectors = einsum('nkc,c->nk', K, corner_pressures)`` (6 components) and the full area.

    ``K`` has shape (n_nodes, 6, n_corner); linear in the corner pressures and computed with the same quadrature
    and pressure interpolation as ``wet_region_load`` (for sign +1).  Also returns ``corner_areas`` (integral of each
    corner hat function) so full-area averages use the same physical integral.
    """
    n_nodes = _check_element(element)
    n_corner = _corner_count(n_nodes)
    ref, triangles = _reference(n_corner)
    points, weights, hats = _full_points(ref, triangles, _rule(element))
    vectors, magnitude = _integrand(element, mesh).evaluate(points)
    phi = _hat_to_corner(hats, n_corner)                       # (M, n_corner)
    kernel = np.einsum('m,mnk,mc->nkc', weights, vectors, phi)
    corner_areas = np.einsum('m,m,mc->c', weights, magnitude, phi)
    _require_native_equivalence(element, mesh, kernel)
    return kernel, float(np.sum(weights * magnitude)), corner_areas


def _require_native_equivalence(element, mesh, kernel):
    """All-component evidence that the interpolation IS the element's own uniform dead-pressure load.

    Compares every nodal component (translations AND moments).  A formulation whose native load has rotational
    terms but whose region kernel does not carry them is refused rather than silently dropping work.
    """
    native = getattr(element, 'compute_native_dead_pressure_load', None)
    if not callable(native):
        return
    expected = np.asarray(native(mesh, 1.0), dtype=float).reshape(len(element.node_ids), -1)
    actual = kernel.sum(axis=2)
    if expected.shape[1] < 6:
        expected = np.hstack([expected, np.zeros((len(expected), 6 - expected.shape[1]))])
    scale = max(float(np.abs(expected).max()), 1e-300)
    if not np.allclose(actual, expected[:, :6], rtol=1e-9, atol=1e-9 * scale):
        raise PressureRegionError('shell formulation pressure interpolation differs from its shape-function integral')


def wet_region_load(element, mesh, corner_levels, corner_pressures, *, sign=1.0, clipper=None, total_area=None):
    """Equivalent nodal vectors for ``p`` on the part of ``element`` where ``level > 0``.

    ``corner_levels``/``corner_pressures`` have one value per corner node (3 or 4, in element node order); the
    pressure value is the continuous continuation, so it reaches zero on the clipped boundary when it is a
    hydrostatic head above the surface.  ``sign`` (+1/-1) is the caller's fluid-side orientation of the pressure
    relative to the element's positive pressure direction.  ``total_area`` may be supplied from ``element_kernel``.
    """
    n_nodes = _check_element(element)
    n_corner = _corner_count(n_nodes)
    levels = np.asarray(corner_levels, dtype=float).reshape(-1)
    pressures = np.asarray(corner_pressures, dtype=float).reshape(-1)
    if levels.shape != (n_corner,) or pressures.shape != (n_corner,) or not np.isfinite(levels).all() \
            or not np.isfinite(pressures).all() or sign not in (-1.0, 1.0):
        raise PressureRegionError('finite corner level and pressure values with a +/-1 orientation are required')
    clip = clipper or _default_clipper()
    integrand = _integrand(element, mesh)
    ref, triangles = _reference(n_corner)
    node_levels = levels if n_corner == 3 else np.r_[levels, levels.mean()]
    node_pressures = pressures if n_corner == 3 else np.r_[pressures, pressures.mean()]
    rp, rw = _rule(element)
    bary_rule = np.column_stack([1. - rp[:, 0] - rp[:, 1], rp[:, 0], rp[:, 1]])
    points, weights, point_pressure = [], [], []
    any_partial = False
    for tri in triangles:
        vertices = ref[list(tri)]
        result = clip(node_levels[list(tri)])
        if result.state == 'dry':
            continue
        any_partial = any_partial or result.state == 'partial'
        bary = np.asarray(result.parameters, dtype=float)
        for a, b, c in result.fan:
            region = bary[[a, b, c]] @ vertices                          # reference coordinates (3, 2)
            region_pressure = bary[[a, b, c]] @ node_pressures[list(tri)]
            area_ref = .5 * abs(_det2(region[1] - region[0], region[2] - region[0]))
            if area_ref <= 0.:
                continue
            points.append(bary_rule @ region)
            weights.append(2. * area_ref * rw)
            point_pressure.append(bary_rule @ region_pressure)
    if total_area is None:
        full_points, full_weights, _ = _full_points(ref, triangles, (rp, rw))
        total_area = float(np.sum(full_weights * integrand.evaluate(full_points)[1]))
    node_ids = tuple(int(i) for i in element.node_ids)
    if not points:
        return RegionLoad(node_ids, np.zeros((n_nodes, 3)), 0., float(total_area), 0., 0., 'dry', np.zeros((n_nodes, 3)))
    points, weights, point_pressure = np.vstack(points), np.concatenate(weights), np.concatenate(point_pressure)
    vectors, magnitude = integrand.evaluate(points)
    nodal = sign * np.einsum('m,mnk->nk', weights * point_pressure, vectors)
    if not np.isfinite(nodal).all():
        raise PressureRegionError('non-finite equivalent load')
    wet_area = float(np.sum(weights * magnitude))
    fraction = wet_area / total_area if total_area > 0. else 0.
    state = 'wet' if fraction > 1. - 1e-12 and not any_partial else 'partial'
    return RegionLoad(node_ids, nodal[:, :3], wet_area, float(total_area), float(min(1., fraction)),
                      float(np.sum(weights * point_pressure * magnitude) / total_area) if total_area > 0. else 0.,
                      state, nodal[:, 3:])


def region_wrench(load, coordinates, origin=(0., 0., 0.)):
    """(Fx,Fy,Fz,Mx,My,Mz) about ``origin``: direct nodal moments plus r x F, for node ``coordinates`` (n_nodes, 3)."""
    x = np.asarray(coordinates, dtype=float) - np.asarray(origin, dtype=float)
    return np.r_[load.forces.sum(axis=0), np.cross(x, load.forces).sum(axis=0) + load.moments.sum(axis=0)]


def add_region_load(load_case, load):
    """Add the full 6-DOF equivalent vectors through the public ``LoadCase.add_nodal_load``."""
    for node_id, vector in zip(load.node_ids, load.vectors):
        if np.any(vector != 0.):
            load_case.add_nodal_load(node_id, [float(v) for v in vector])
