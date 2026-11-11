"""Exploratory finite shell station pose with independent physical director.

The four material samples form a declared finite-width rail. A director-
constrained planar proper fit supplies its in-plane spin, while the physical
director remains independent of midsurface slopes. This is a bounded G0
candidate, not an admitted E4-PL trial-state adapter or selected line law.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from anyfesolver import _ge_beam3_mixed_ad as ad
from anyfesolver.corotational import rotation_matrix_from_vector

from scripts._b3_e4_f03_shell_spatial_linear_port import reference_spatial_trace
from scripts._b3_e4_f03_shell_trace import ShellPanel, _SAMPLE_OFFSETS


_OFFSETS = np.asarray(_SAMPLE_OFFSETS, dtype=float)
_SECOND_X = float(_OFFSETS[:, 0]@_OFFSETS[:, 0])
_SECOND_Y = float(_OFFSETS[:, 1]@_OFFSETS[:, 1])


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array(((0., -z, y), (z, 0., -x), (-y, x, 0.)))


def reference_nodes(panel: ShellPanel) -> np.ndarray:
    """The flat +Z source mesh used by the bounded F03 shell diagnostics."""
    if (not isinstance(panel, ShellPanel) or panel.topology not in
            ("Q4", "S3-V2D") or not 1 <= panel.n_s <= 8 or
            panel.node_count != (panel.n_y+1)*(2*panel.n_s+1)):
        raise ValueError("bounded flat F03 Q4/S3 panel required")
    nx = 2*panel.n_s
    return np.array([(-.5+i/panel.n_s, y, 0.)
                     for y in np.linspace(-.5, .5, panel.n_y+1)
                     for i in range(nx+1)])


def pose(panel: ShellPanel, station: float, positions: np.ndarray,
         directors: np.ndarray, *,
         _director_weights: np.ndarray | None = None,
         _origin_weights: np.ndarray | None = None,
         _planar_weights: np.ndarray | None = None
         ) -> tuple[np.ndarray, np.ndarray]:
    """Return material-sample centroid and proper finite spatial frame.

    Inputs are current global nodal positions and unit physical directors.
    Director extraction from an E4-PL state is deliberately outside this
    candidate. The fit is smooth only on its checked nondegenerate branch.
    """
    nodes = reference_nodes(panel)
    x = np.asarray(positions, dtype=float)
    d = np.asarray(directors, dtype=float)
    if (x.shape != nodes.shape or d.shape != nodes.shape or
            not np.isfinite(x).all() or not np.isfinite(d).all() or
            np.max(np.abs(np.linalg.norm(d, axis=1)-1.)) > 1e-10):
        raise ValueError("finite current nodes and unit physical directors required")
    weights = (panel.sample_weights(station) if
               (_origin_weights is None or _planar_weights is None or
                _director_weights is None) else None)
    samples = weights@x if weights is not None else None
    if _origin_weights is None:
        centroid = samples.mean(axis=0)
    else:
        origin_weights = np.asarray(_origin_weights, dtype=float)
        if (origin_weights.shape != (panel.node_count,) or
                not np.isfinite(origin_weights).all() or
                abs(float(np.sum(origin_weights))-1.) > 1e-12):
            raise ValueError("finite normalized shell origin weights required")
        centroid = origin_weights@x
    if _planar_weights is None:
        centered = samples-samples.mean(axis=0)
        a = centered.T@_OFFSETS[:, 0]
        b = centered.T@_OFFSETS[:, 1]
        scale_x, scale_y = _SECOND_X, _SECOND_Y
    else:
        planar_weights = np.asarray(_planar_weights, dtype=float)
        if (planar_weights.shape != (2, panel.node_count) or
                not np.isfinite(planar_weights).all() or
                np.max(np.abs(np.sum(planar_weights, axis=1))) > 1e-12):
            raise ValueError("finite shell material-gradient weights required")
        a, b = planar_weights@x
        scale_x = scale_y = 1.
    director_weights = (weights.mean(axis=0) if _director_weights is None else
                        np.asarray(_director_weights, dtype=float))
    if (director_weights.shape != (panel.node_count,) or
            not np.isfinite(director_weights).all() or
            abs(float(np.sum(director_weights))-1.) > 1e-12):
        raise ValueError("finite normalized shell director weights required")
    director = director_weights@d
    length = float(np.linalg.norm(director))
    if not np.isfinite(length) or length <= 1e-8:
        raise ValueError("cancelled shell director field")
    director /= length
    cosine = float(director[2])
    if cosine <= -.95:
        raise ValueError("antipodal shell director chart")
    v = np.cross((0., 0., 1.), director)
    skew = _skew(v)
    chart = np.eye(3)+skew+(skew@skew)/(1.+cosine)
    projected = chart[:, :2].T@np.column_stack((a, b))
    signed_area = float(np.linalg.det(projected))
    if (not np.isfinite(signed_area) or
            signed_area <= 1e-8*scale_x*scale_y):
        raise ValueError("degenerate material rail orientation")
    trace = float(projected[0, 0]+projected[1, 1])
    spin = float(projected[1, 0]-projected[0, 1])
    norm = float(np.hypot(trace, spin))
    if not np.isfinite(norm) or norm <= 1e-8*(scale_x+scale_y):
        raise ValueError("singular material rail planar fit")
    c, s = trace/norm, spin/norm
    frame = chart@np.array(((c, -s, 0.), (s, c, 0.), (0., 0., 1.)))
    if (not np.isfinite(frame).all() or
            np.linalg.norm(frame.T@frame-np.eye(3)) > 2e-12 or
            np.linalg.det(frame) <= 0.):
        raise ValueError("invalid finite shell spatial frame")
    return centroid.copy(), frame.copy()


def reference_tangent(panel: ShellPanel, station: float) -> np.ndarray:
    """Existing independently assembled flat-reference linearization oracle."""
    return reference_spatial_trace(panel, station)


def pose_from_e4_total(panel: ShellPanel, station: float,
                       total_u: np.ndarray, *,
                       _director_weights: np.ndarray | None = None,
                       _origin_weights: np.ndarray | None = None,
                       _planar_weights: np.ndarray | None = None
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate this candidate from flat-panel E4 additive total coordinates.

    This mirrors the nodal ``Exp(total_u[3:6])`` used by the existing
    corotational shell pullback. It does not authenticate an E4 trial or its
    material predecessor, and is limited to this flat +Z panel.
    """
    nodes = reference_nodes(panel)
    values = np.asarray(total_u, dtype=float)
    if values.shape != (6*panel.node_count,) or not np.isfinite(values).all():
        raise ValueError("finite complete flat E4 total coordinates required")
    nodal = values.reshape(-1, 6)
    positions = nodes+nodal[:, :3]
    directors = np.array([rotation_matrix_from_vector(theta)[:, 2]
                          for theta in nodal[:, 3:]])
    return pose(panel, station, positions, directors,
                _director_weights=_director_weights,
                _origin_weights=_origin_weights,
                _planar_weights=_planar_weights)


@dataclass(frozen=True)
class DirectionalPose:
    """Owned first and mixed second spatial-pose matrix variations."""

    position_first: np.ndarray
    position_second: np.ndarray
    position_mixed: np.ndarray
    frame_first: np.ndarray
    frame_second: np.ndarray
    frame_mixed: np.ndarray


def directional_pose(panel: ShellPanel, station: float,
                     positions: np.ndarray, directors: np.ndarray,
                     first: np.ndarray, second: np.ndarray, *,
                     _total_rotations: np.ndarray | None = None,
                     _director_weights: np.ndarray | None = None,
                     _origin_weights: np.ndarray | None = None,
                     _planar_weights: np.ndarray | None = None) -> DirectionalPose:
    """Differentiate the finite port in two actual spatial-left directions.

    Each nodal director follows ``Exp(epsilon*w1+eta*w2) @ director``.
    This is a physical spatial variation, not the Hessian of the
    relative-pose Log constraint. The internal total-rotation option below
    is used only by the explicit additive-E4 adapter.
    """
    x, d = np.asarray(positions, dtype=float), np.asarray(directors, dtype=float)
    u, v = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    pose(panel, station, x, d,
         _director_weights=_director_weights,
         _origin_weights=_origin_weights,
         _planar_weights=_planar_weights)
    if (u.shape != (panel.node_count, 6) or
            v.shape != (panel.node_count, 6) or
            not np.isfinite(u).all() or not np.isfinite(v).all()):
        raise ValueError("two finite six-channel nodal directions required")
    zero = ad.Jet2.constant(0., 2)
    epsilon, eta = ad.Jet2.variable(0., 0, 2), ad.Jet2.variable(0., 1, 2)
    nodes = []
    director_jets = []
    for i in range(panel.node_count):
        nodes.append([ad.Jet2.constant(x[i, j], 2)+
                      epsilon*u[i, j]+eta*v[i, j] for j in range(3)])
        spin = [epsilon*u[i, j+3]+eta*v[i, j+3] for j in range(3)]
        if _total_rotations is None:
            origin = [ad.Jet2.constant(d[i, j], 2) for j in range(3)]
        else:
            spin = [ad.Jet2.constant(_total_rotations[i, j], 2)+spin[j]
                    for j in range(3)]
            origin = [zero, zero, 1.+zero]
        director_jets.append(ad.matvec(ad.so3_exp(spin), origin))
    weights = (panel.sample_weights(station) if
               (_origin_weights is None or _planar_weights is None or
                _director_weights is None) else None)
    samples = ([[sum((weights[k, i]*nodes[i][j]
                     for i in range(panel.node_count)), zero)
                 for j in range(3)] for k in range(4)]
               if weights is not None else None)
    if _origin_weights is None:
        center = [sum((sample[j] for sample in samples), zero)/4.
                  for j in range(3)]
    else:
        center = [sum((_origin_weights[i]*nodes[i][j]
                       for i in range(panel.node_count)), zero)
                  for j in range(3)]
    if _planar_weights is None:
        sample_center = [sum((sample[j] for sample in samples), zero)/4.
                         for j in range(3)]
        a = [sum(((samples[k][j]-sample_center[j])*_OFFSETS[k, 0]
                  for k in range(4)), zero) for j in range(3)]
        b = [sum(((samples[k][j]-sample_center[j])*_OFFSETS[k, 1]
                  for k in range(4)), zero) for j in range(3)]
    else:
        a = [sum((_planar_weights[0, i]*nodes[i][j]
                  for i in range(panel.node_count)), zero)
             for j in range(3)]
        b = [sum((_planar_weights[1, i]*nodes[i][j]
                  for i in range(panel.node_count)), zero)
             for j in range(3)]
    director_weights = (weights.mean(axis=0) if _director_weights is None else
                        np.asarray(_director_weights, dtype=float))
    director_sum = [sum((director_weights[i]*director_jets[i][j]
                         for i in range(panel.node_count)), zero)
                    for j in range(3)]
    director_length = ad.sqrt(ad.dot(director_sum, director_sum))
    normal = [entry/director_length for entry in director_sum]
    spin = [-normal[1], normal[0], zero]
    skew = ad.skew(spin)
    skew2 = ad.matmul(skew, skew)
    chart = [[(1. if i == j else 0.)+skew[i][j]+
              skew2[i][j]/(1.+normal[2]) for j in range(3)]
             for i in range(3)]
    projected = [[ad.dot([chart[k][i] for k in range(3)], axis)
                  for axis in (a, b)] for i in range(2)]
    trace = projected[0][0]+projected[1][1]
    spin = projected[1][0]-projected[0][1]
    length = ad.sqrt(trace*trace+spin*spin)
    c, s = trace/length, spin/length
    yaw = [[c, -s, zero], [s, c, zero], [zero, zero, 1.+zero]]
    frame = ad.matmul(chart, yaw)
    numeric_center, numeric_frame = pose(
        panel, station, x, d,
        _director_weights=_director_weights,
        _origin_weights=_origin_weights,
        _planar_weights=_planar_weights)
    if (np.max(np.abs([entry.value for entry in center]-numeric_center)) > 1e-11 or
            np.max(np.abs([[entry.value for entry in row] for row in frame]-
                          numeric_frame)) > 1e-11):
        raise AssertionError("finite shell pose and derivative charts disagree")
    def vector(order):
        return np.array([entry.gradient[order] for entry in center])
    def matrix(order):
        return np.array([[entry.gradient[order] for entry in row]
                         for row in frame])
    return DirectionalPose(
        vector(0), vector(1),
        np.array([entry.hessian[0, 1] for entry in center]),
        matrix(0), matrix(1),
        np.array([[entry.hessian[0, 1] for entry in row]
                  for row in frame]))


def directional_pose_from_e4_total(
        panel: ShellPanel, station: float, total_u: np.ndarray,
        first: np.ndarray, second: np.ndarray, *,
        _director_weights: np.ndarray | None = None,
        _origin_weights: np.ndarray | None = None,
        _planar_weights: np.ndarray | None = None) -> DirectionalPose:
    """Pull this candidate through the existing additive total-rotation chart.

    The state-owner authority and constitutive trial remain outside this
    diagnostic. A physical drill is not a single ``rz`` coordinate column
    once the director tilts; the full chart derivative is retained here.
    """
    nodes = reference_nodes(panel)
    values = np.asarray(total_u, dtype=float)
    if values.shape != (6*panel.node_count,) or not np.isfinite(values).all():
        raise ValueError("finite complete flat E4 total coordinates required")
    nodal = values.reshape(-1, 6)
    positions = nodes+nodal[:, :3]
    directors = np.array([rotation_matrix_from_vector(theta)[:, 2]
                          for theta in nodal[:, 3:]])
    return directional_pose(panel, station, positions, directors,
                            first, second, _total_rotations=nodal[:, 3:],
                            _director_weights=_director_weights,
                            _origin_weights=_origin_weights,
                            _planar_weights=_planar_weights)
