"""Exact fixed-end load bubbles for the existing physical linear B2 law.

This private experimental helper does not change the element or panel equations.
Local pose is [u,v,w,rx,ry,rz], wrench is [N,Vy,Vz,T,My,Mz], and strain is
[u',v'-rz,w'+ry,rx',ry',rz']. With external force/couple (p,c) per reference
length, n'=-p and m'=-ex cross n-c. A concentrated external axis wrench (P,Q)
gives right-minus-left jumps (-P,-Q). A port offset d maps its wrench to the
axis by (P,Q+d cross P), and maps axis pose to the port by (r+theta cross d,
theta); these are transpose work maps in m/rad and N/Nm.

For zero nodal coordinates, the existing homogeneous trace J gives equivalent
nodal load f=integral(J.T*w) + sum(J(a).T*W_axis), end tractions -f, and initial
section resultant f[:6]. Integrating compliance times the resulting piecewise
equilibrium field gives the load bubble. Its pose is continuous at point loads
and exactly zero at both endpoints. Endpoint point loads therefore have nodal
work but no fixed-end bubble. All integrations use the actual stored binary64
inputs as Fractions; no quadrature, fitted stiffness or physical floor is used.

The returned point Green matrix is source wrench to source-port bubble pose.
It is reciprocal and nonnegative in work. In a future coupled owner, if A maps
axis pose to attachment constraint channels, W=-A.T*lambda implies
g=C*u+a_body-H*lambda, H_ij=A_i*G(s_i,s_j)*A_j.T. Thus the condensed KKT lower
block is -H and lower RHS is -C*u0-a_body. This helper alone does not implement
that policy, loaded attachment compatibility, loaded energy or qualification.
"""

from dataclasses import dataclass, field
from fractions import Fraction
import math
from numbers import Real

import numpy as np

from ._b2_physical_flexibility_linear import (
    LinearOperator, RepresentabilityError, _array, _multiply,
)


POLICY_ID = "B2_FIXED_END_LOAD_BUBBLE_MAPS_V1_DEV"
_ZERO = Fraction(0)


def _real(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"finite real {label} required")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"finite real {label} required") from exc
    if not math.isfinite(result):
        raise ValueError(f"finite real {label} required")
    return result


def _stations(value, label):
    try:
        rows = np.asarray(value, dtype=object)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"normalized {label} required") from exc
    if rows.ndim != 1:
        raise ValueError(f"normalized {label} required")
    result = tuple(_real(v, label) for v in rows)
    if any(not 0. <= v <= 1. for v in result):
        raise ValueError(f"normalized {label} in [0,1] required")
    return result


def _rows(value, count, width, label):
    if value is None:
        return tuple((0.,)*width for _ in range(count))
    try:
        rows = np.asarray(value, dtype=object)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{count} finite {label} rows required") from exc
    if count == 0 and rows.shape == (0,):
        rows = rows.reshape(0, width)
    if rows.shape != (count, width):
        raise ValueError(f"{count} finite {label} rows required")
    return tuple(tuple(_real(v, label) for v in row) for row in rows)


def _owned(values, shape):
    converted = _array(np.asarray(values, dtype=object).reshape(shape))
    return np.frombuffer(converted.tobytes(), dtype=float).reshape(shape)


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0])


def _axis_wrench(port_wrench, offset):
    moment = _cross(offset, port_wrench[:3])
    return tuple(port_wrench[:3]) + tuple(port_wrench[i+3]+moment[i] for i in range(3))


def _port_pose(axis_pose, offset):
    shift = _cross(axis_pose[3:], offset)
    return tuple(axis_pose[i]+shift[i] for i in range(3)) + tuple(axis_pose[3:])


def _poly_value(poly, at):
    return sum(coefficient*at**degree for degree, coefficient in enumerate(poly))


def _strain_polynomials(compliance, resultants):
    return tuple(tuple(sum(compliance[row][channel]*resultants[channel][degree]
                           for channel in range(6)) for degree in range(3))
                 for row in range(6))


def _integrated_pose(strains, distance):
    ordinary = tuple(sum(c*distance**(degree+1)/Fraction(degree+1)
                         for degree, c in enumerate(poly)) for poly in strains)
    weighted = tuple(sum(c*distance**(degree+2)/Fraction((degree+1)*(degree+2))
                         for degree, c in enumerate(poly)) for poly in strains)
    return (ordinary[0], ordinary[1]+weighted[5], ordinary[2]-weighted[4],
            ordinary[3], ordinary[4], ordinary[5])


def _uniform_strains(compliance, left, wrench):
    ex = (Fraction(1), _ZERO, _ZERO)
    turn, load_turn = _cross(ex, left[:3]), _cross(ex, wrench[:3])
    resultants = tuple((left[i], -wrench[i], _ZERO) for i in range(3)) + tuple(
        (left[i+3], -turn[i]-wrench[i+3], load_turn[i]/2) for i in range(3))
    return _strain_polynomials(compliance, resultants)


def _point_delta_strains(compliance, wrench):
    turn = _cross((Fraction(1), _ZERO, _ZERO), wrench[:3])
    resultants = tuple((-wrench[i], _ZERO, _ZERO) for i in range(3)) + tuple(
        (-wrench[i+3], turn[i], _ZERO) for i in range(3))
    return _strain_polynomials(compliance, resultants)


def _point_equivalent(trace_polynomials, station, wrench):
    return tuple(sum(_poly_value(trace_polynomials[row][column], station)*wrench[row]
                     for row in range(6)) for column in range(12))


def _axis_bubble(length, compliance, equivalent, uniform_wrench, station,
                 point_station=None, point_wrench=None):
    distance = length*station
    base = _uniform_strains(compliance, equivalent[:6], uniform_wrench)
    pose = _integrated_pose(base, distance)
    if point_station is not None and station > point_station:
        delta = _integrated_pose(_point_delta_strains(compliance, point_wrench),
                                 length*(station-point_station))
        pose = tuple(a+b for a, b in zip(pose, delta))
    return pose


@dataclass(frozen=True, eq=False)
class LocalLoadBubbleResponse:
    stations: tuple[float, ...]
    pose: np.ndarray
    equivalent_nodal_load: np.ndarray
    policy_id: str = field(default=POLICY_ID, init=False)
    production_qualified: bool = field(default=False, init=False)

    def __post_init__(self):
        object.__setattr__(self, "stations", _stations(self.stations, "response stations"))
        for name, shape in (("pose", (len(self.stations), 6)),
                            ("equivalent_nodal_load", (12,))):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != shape or not np.isfinite(value).all():
                raise RepresentabilityError("fixed-end bubble output is not representable")
            object.__setattr__(self, name,
                np.frombuffer(value.tobytes(), dtype=float).reshape(shape))


@dataclass(frozen=True, eq=False, init=False)
class FixedEndLoadBubbleMaps:
    """Compile local maps for selected observations and declared point ports.

    Uniform wrench is per reference length; point wrenches are total actions
    about their declared port offsets. Observation offsets are rigid offsets
    from the axis at each selected station. Empty observations/sources and
    repeated stations are admitted; source order is preserved. Point compliance
    always observes each source at its own offset, regardless of the selected
    observation list. Float arrays are owned bytes-backed snapshots. Evaluation
    uses retained exact maps, avoiding loss from intermediate float products.
    """

    stations: tuple[float, ...]
    point_stations: tuple[float, ...]
    observation_offsets_local_m: tuple[tuple[float, ...], ...]
    point_offsets_local_m: tuple[tuple[float, ...], ...]
    uniform_pose_map: np.ndarray
    point_pose_map: np.ndarray
    point_compliance: np.ndarray
    uniform_equivalent_load_map: np.ndarray
    point_equivalent_load_map: np.ndarray
    policy_id: str
    production_qualified: bool
    _uniform_exact: tuple = field(repr=False)
    _point_exact: tuple = field(repr=False)
    _uniform_equivalent_exact: tuple = field(repr=False)
    _point_equivalent_exact: tuple = field(repr=False)

    def __init__(self, operator, *, stations=(0., .5, 1.), point_stations=(),
                 observation_offsets_local_m=None, point_offsets_local_m=None):
        if type(operator) is not LinearOperator:
            raise ValueError("existing exact physical B2 LinearOperator required")
        selected = _stations(stations, "observation stations")
        sources = _stations(point_stations, "point stations")
        offsets = _rows(observation_offsets_local_m, len(selected), 3, "observation offset")
        source_offsets = _rows(point_offsets_local_m, len(sources), 3, "point offset")
        x = tuple(Fraction(v) for v in selected)
        a = tuple(Fraction(v) for v in sources)
        d = tuple(tuple(Fraction(v) for v in row) for row in offsets)
        source_d = tuple(tuple(Fraction(v) for v in row) for row in source_offsets)
        length, compliance = operator.length, operator._compliance
        trace = operator._trace_polynomials
        basis = tuple(tuple(Fraction(int(i == j)) for i in range(6)) for j in range(6))
        zero = (_ZERO,)*6
        uniform_equivalent = tuple(tuple(row) for row in operator._uniform_load_map)
        # Store matrices with output channel before input channel.
        point_equivalent = tuple(tuple(
            _point_equivalent(trace, at, _axis_wrench(unit, offset)) for unit in basis)
            for at, offset in zip(a, source_d))

        def uniform_pose(at, offset):
            columns = tuple(_port_pose(_axis_bubble(length, compliance,
                _multiply(uniform_equivalent, unit), unit, at), offset) for unit in basis)
            return tuple(tuple(columns[column][row] for column in range(6)) for row in range(6))

        def point_pose(at, offset, source_index):
            columns = tuple(_port_pose(_axis_bubble(length, compliance,
                point_equivalent[source_index][column], zero, at, a[source_index],
                _axis_wrench(unit, source_d[source_index])), offset)
                for column, unit in enumerate(basis))
            return tuple(tuple(columns[column][row] for column in range(6)) for row in range(6))

        # Certify exact fixed-end closure independently of observation selection.
        for at in (Fraction(0), Fraction(1)):
            if any(value != 0 for row in uniform_pose(at, zero) for value in row):
                raise RepresentabilityError("exact uniform load-bubble endpoint closure failed")
            for source_index in range(len(a)):
                if any(value != 0 for row in point_pose(at, zero, source_index) for value in row):
                    raise RepresentabilityError("exact point load-bubble endpoint closure failed")
        uniform = tuple(uniform_pose(at, offset) for at, offset in zip(x, d))
        point = tuple(tuple(point_pose(at, offset, j) for j in range(len(a)))
                      for at, offset in zip(x, d))
        blocks = tuple(tuple(point_pose(at, offset, j) for j in range(len(a)))
                       for at, offset in zip(a, source_d))
        green = tuple(tuple(blocks[i][j][row][column]
                            for j in range(len(a)) for column in range(6))
                      for i in range(len(a)) for row in range(6))
        if any(green[i][j] != green[j][i] for i in range(6*len(a)) for j in range(i)):
            raise RepresentabilityError("exact fixed-end Green reciprocity failed")
        point_equivalent_rows = tuple(tuple(tuple(point_equivalent[j][column][row]
            for column in range(6)) for j in range(len(a))) for row in range(12))
        fields = dict(stations=selected, point_stations=sources,
            observation_offsets_local_m=offsets, point_offsets_local_m=source_offsets,
            uniform_pose_map=_owned(uniform, (len(x), 6, 6)),
            point_pose_map=_owned(point, (len(x), len(a), 6, 6)),
            point_compliance=_owned(green, (6*len(a), 6*len(a))),
            uniform_equivalent_load_map=_owned(uniform_equivalent, (12, 6)),
            point_equivalent_load_map=_owned(point_equivalent_rows, (12, len(a), 6)),
            policy_id=POLICY_ID, production_qualified=False,
            _uniform_exact=uniform, _point_exact=point,
            _uniform_equivalent_exact=uniform_equivalent,
            _point_equivalent_exact=point_equivalent_rows)
        for name, value in fields.items():
            object.__setattr__(self, name, value)

    def evaluate(self, uniform_wrench=(0.,)*6, *, point_wrenches=None):
        """Apply exact compiled maps; return bubbles, not homogeneous nodal pose."""
        uniform = _rows((uniform_wrench,), 1, 6, "uniform wrench")[0]
        points = _rows(point_wrenches, len(self.point_stations), 6, "point wrench")
        w = tuple(Fraction(v) for v in uniform)
        p = tuple(tuple(Fraction(v) for v in row) for row in points)
        pose = tuple(tuple(sum(self._uniform_exact[i][row][column]*w[column] for column in range(6))
            + sum(self._point_exact[i][j][row][column]*p[j][column]
                  for j in range(len(p)) for column in range(6)) for row in range(6))
            for i in range(len(self.stations)))
        equivalent = tuple(sum(self._uniform_equivalent_exact[row][column]*w[column] for column in range(6))
            + sum(self._point_equivalent_exact[row][j][column]*p[j][column]
                  for j in range(len(p)) for column in range(6)) for row in range(12))
        return LocalLoadBubbleResponse(self.stations, _owned(pose, (len(self.stations), 6)),
                                       _owned(equivalent, (12,)))
