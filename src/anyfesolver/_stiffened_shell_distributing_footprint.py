"""Private flat-Q4 distributing footprint reference-linear work compiler.

The frozen area mean/director/polar-yaw trace preserves the existing bounded
prototype. Rows are declared common reference-spatial rows, not arbitrary raw
Log-coordinate rows. No finite pose, chart, public admission or qualification
is implemented. Native beam elements and retained variables are unchanged.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from hashlib import sha256
from math import isfinite
from numbers import Real
import json

import numpy as np


LAW = "GE_FLAT_Q4_DISTRIBUTING_DIRECTOR_POLAR_REFERENCE_LINEAR_V1_EXPERIMENTAL"
IDENTITY_AXES = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
MAX_HOSTS = 128
MAX_SHELL_NODES = 256


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _identifier(value):
    _require(type(value) is str and value and value == value.strip(), "explicit footprint ID required")


def _node_id(value):
    _require(type(value) is int and value > 0, "positive plain-int model ID required")


def _offset(value):
    _require(type(value) is int and 0 <= value <= np.iinfo(np.int64).max-5 and value % 6 == 0,
             "explicit physical six-coordinate q offset required")


def _numbers(values, size):
    try:
        owned = tuple(values)
    except TypeError as error:
        raise ValueError("complete finite real coordinates required") from error
    _require(len(owned) == size and all(isinstance(v, Real) and not isinstance(v, (bool, np.bool_))
                                      and isfinite(v) for v in owned),
             "complete finite real coordinates required")
    return tuple(float(v) for v in owned)


def _readonly(values, dtype=float):
    array = np.asarray(values, dtype=dtype)
    # Immutable bytes backing also prevents callers from reenabling writes.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _vector(values, size):
    array = np.asarray(values)
    _require(array.shape == (size,) and array.dtype.kind in "fiu" and np.isfinite(array).all(),
             "complete finite real physical-coordinate vector required")
    owned = np.asarray(array, dtype=float)
    _require(np.isfinite(owned).all(), "representable physical coordinates required")
    return owned


@dataclass(frozen=True)
class FlatQ4Node:
    """One actual shell node: ID, reference position and physical u/omega offset."""
    node_id: int
    coordinates_m: tuple[float, float, float]
    q_offset: int

    def __post_init__(self):
        _node_id(self.node_id)
        _offset(self.q_offset)
        object.__setattr__(self, "coordinates_m", _numbers(self.coordinates_m, 3))


@dataclass(frozen=True)
class CartesianQ4Host:
    """Actual element ID and its CCW cyclic four-node incidence."""
    element_id: int
    node_ids: tuple[int, int, int, int]

    def __post_init__(self):
        _node_id(self.element_id)
        ids = tuple(self.node_ids)
        _require(len(ids) == 4 and len(set(ids)) == 4, "four distinct Q4 model node IDs required")
        for value in ids:
            _node_id(value)
        object.__setattr__(self, "node_ids", ids)


@dataclass(frozen=True)
class PhysicalBeamNodePort:
    """Existing physical beam node's six common reference-spatial coordinates.

    The explicit identity reference axes are mandatory. This is not an element
    interpolation, recovered station pose or native retained-variable port.
    """
    node_id: int
    coordinates_m: tuple[float, float, float]
    q_offset: int
    reference_axes: tuple[tuple[float, float, float], ...]

    def __post_init__(self):
        _node_id(self.node_id)
        _offset(self.q_offset)
        object.__setattr__(self, "coordinates_m", _numbers(self.coordinates_m, 3))
        axes = tuple(_numbers(row, 3) for row in self.reference_axes)
        _require(axes == IDENTITY_AXES, "common identity reference-spatial axes required")
        object.__setattr__(self, "reference_axes", axes)


@dataclass(frozen=True)
class DistributingFootprintDefinition:
    footprint_id: str
    host_element_ids: tuple[int, ...]
    beam_node_id: int
    bounds_xy_m: tuple[float, float, float, float]  # xmin,xmax,ymin,ymax

    def __post_init__(self):
        _identifier(self.footprint_id)
        _node_id(self.beam_node_id)
        ids = tuple(self.host_element_ids)
        _require(2 <= len(ids) <= MAX_HOSTS and len(set(ids)) == len(ids),
                 "two to 128 distinct host element IDs required")
        for value in ids:
            _node_id(value)
        bounds = _numbers(self.bounds_xy_m, 4)
        _require(bounds[0] < bounds[1] and bounds[2] < bounds[3], "positive rectangular footprint area required")
        object.__setattr__(self, "host_element_ids", ids)
        object.__setattr__(self, "bounds_xy_m", bounds)


@dataclass(frozen=True, eq=False)
class CompiledDistributingFootprint:
    definition: DistributingFootprintDefinition
    shell_node_ids: tuple[int, ...]
    host_element_ids: tuple[int, ...]
    beam_node_id: int
    gather_slots: np.ndarray
    shell_positions_m: np.ndarray
    beam_position_m: np.ndarray
    centroid_m: np.ndarray
    eccentricity_m: np.ndarray
    area_m2: float
    area_mean: np.ndarray
    planar_gradients: np.ndarray
    shell_trace: np.ndarray
    constraint_matrix: np.ndarray
    identity: str

    def __post_init__(self):
        _require(type(self.definition) is DistributingFootprintDefinition, "typed compiled definition required")
        node_ids, host_ids = tuple(self.shell_node_ids), tuple(self.host_element_ids)
        count = len(node_ids)
        _require(6 <= count <= MAX_SHELL_NODES and count == len(set(node_ids)) and
                 2 <= len(host_ids) <= MAX_HOSTS and len(host_ids) == len(set(host_ids)),
                 "complete distinct compiled model IDs required")
        for value in node_ids+host_ids+(self.beam_node_id,):
            _node_id(value)
        _require(set(host_ids) == set(self.definition.host_element_ids) and
                 self.beam_node_id == self.definition.beam_node_id, "compiled model binding differs")
        object.__setattr__(self, "shell_node_ids", node_ids)
        object.__setattr__(self, "host_element_ids", host_ids)
        shapes = dict(shell_positions_m=(count, 3), beam_position_m=(3,), centroid_m=(3,),
                      eccentricity_m=(3,), area_mean=(count,), planar_gradients=(2, count),
                      shell_trace=(6, 6*count), constraint_matrix=(6, 6*(count+1)))
        for name, shape in shapes.items():
            array = np.asarray(getattr(self, name))
            _require(array.shape == shape and array.dtype.kind in "fiu" and np.isfinite(array).all(),
                     "complete finite real compiled local map required")
            owned = _readonly(array)
            _require(np.isfinite(owned).all(), "representable compiled local map required")
            object.__setattr__(self, name, owned)
        slots = np.asarray(self.gather_slots)
        _require(slots.shape == (6*(count+1),) and slots.dtype.kind in "iu" and
                 np.all(slots >= 0) and np.all(slots <= np.iinfo(np.int64).max) and
                 len(set(slots.tolist())) == 6*(count+1), "distinct explicit compiled physical gather required")
        for block in slots.reshape(count+1, 6):
            _offset(int(block[0]))
            _require(np.array_equal(block, int(block[0])+np.arange(6)), "physical six-coordinate gather block required")
        object.__setattr__(self, "gather_slots", _readonly(slots, np.int64))
        _require(isinstance(self.area_m2, Real) and isfinite(self.area_m2) and self.area_m2 > 0.,
                 "positive finite compiled area required")
        _require(type(self.identity) is str and len(self.identity) == 64 and
                 all(c in "0123456789abcdef" for c in self.identity), "compiled binding hash required")

    def gather(self, physical_coordinates):
        """Own just the explicitly mapped nodal coordinates; no global matrix."""
        array = np.asarray(physical_coordinates)
        _require(array.ndim == 1 and array.dtype.kind in "fiu" and
                 len(array) > int(self.gather_slots.max()), "complete physical nodal gather input required")
        return _readonly(_vector(array[self.gather_slots], len(self.gather_slots)))

    def gap(self, local_coordinates):
        with np.errstate(over="ignore", invalid="ignore"):
            result = self.constraint_matrix @ _vector(local_coordinates, len(self.gather_slots))
        _require(np.isfinite(result).all(), "representable local gap required")
        return _readonly(result)

    def transpose_work(self, multipliers):
        """Local shell/beam resisting covectors; exerted covectors have minus sign."""
        with np.errstate(over="ignore", invalid="ignore"):
            result = self.constraint_matrix.T @ _vector(multipliers, 6)
        _require(np.isfinite(result).all(), "representable local work covectors required")
        return _readonly(result)

    def descriptor(self):
        return dict(law=LAW, footprint_id=self.definition.footprint_id,
                    host_element_ids=list(self.host_element_ids), shell_node_ids=list(self.shell_node_ids),
                    beam_node_id=self.beam_node_id, bounds_xy_m=list(self.definition.bounds_xy_m),
                    reference_rows="common identity reference-spatial u/omega",
                    spin="polar yaw from planar displacement gradients; no shell rz action",
                    configuration="reference_linear", public_admission=False, production_qualified=False,
                    identity=self.identity)


def _host(host, nodes):
    _require(all(n in nodes for n in host.node_ids), "host references unknown shell node")
    positions = [nodes[n].coordinates_m for n in host.node_ids]
    xs, ys = sorted({p[0] for p in positions}), sorted({p[1] for p in positions})
    _require(len(xs) == len(ys) == 2 and len({p[2] for p in positions}) == 1,
             "flat Cartesian identity-XY Q4 hosts required")
    corners = [(xs[0], ys[0]), (xs[1], ys[0]), (xs[1], ys[1]), (xs[0], ys[1])]
    incidence = [(p[0], p[1]) for p in positions]
    _require(any(incidence == corners[k:]+corners[:k] for k in range(4)),
             "CCW cyclic Cartesian Q4 host incidence required")
    ordered = tuple(host.node_ids[incidence.index(c)] for c in corners)
    return ordered, tuple(map(F, (xs[0], xs[1], ys[0], ys[1]))), F(positions[0][2])


def _one_dimensional(left, right, low, high):
    width, interval = high-low, right-left
    first = (high*interval-(right*right-left*left)/2)/width
    second = ((right*right-left*left)/2-low*interval)/width
    return (first, second), (-interval/width, interval/width)


def _rectangular_hosts(geometry):
    """Bounded exact tensor-grid coverage and actual shared-edge identities."""
    xs = sorted({value for _, bounds, _ in geometry for value in bounds[:2]})
    ys = sorted({value for _, bounds, _ in geometry for value in bounds[2:]})
    xi, yi = {v: i for i, v in enumerate(xs)}, {v: i for i, v in enumerate(ys)}
    tiles = {}
    for ids, (x0, x1, y0, y1), _ in geometry:
        _require(xi[x1] == xi[x0]+1 and yi[y1] == yi[y0]+1,
                 "conforming Cartesian host edges required; coarse/fine traces unsupported")
        key = xi[x0], yi[y0]
        _require(key not in tiles, "overlapping Cartesian host coverage")
        tiles[key] = ids
    _require(len(tiles) == (len(xs)-1)*(len(ys)-1),
             "complete rectangular Cartesian host grid required; holes/disconnected hosts unsupported")
    for (ix, iy), ids in tiles.items():
        right, top = tiles.get((ix+1, iy)), tiles.get((ix, iy+1))
        if right is not None:
            _require({ids[1], ids[2]} == {right[0], right[3]},
                     "conforming shared host edge-node identities required")
        if top is not None:
            _require({ids[2], ids[3]} == {top[0], top[1]},
                     "conforming shared host edge-node identities required")


def compile_footprint(definition, nodes, hosts, beam_port):
    """Compile 2..128 conforming rectangular Q4 hosts and one physical beam node.

    Integrals use exact rational arithmetic on stored reference coordinates,
    then one binary64 conversion. This is a derivative/work contract for the
    selected area-mean/director/polar-yaw prototype, not a finite pose fit.
    """
    _require(type(definition) is DistributingFootprintDefinition and type(beam_port) is PhysicalBeamNodePort,
             "typed distributing definition and physical beam nodal port required")
    nodes, hosts = tuple(nodes), tuple(hosts)
    count = len(nodes)
    _require(6 <= count <= MAX_SHELL_NODES and all(type(n) is FlatQ4Node for n in nodes),
             "bounded complete local shell inventory required (6..256 nodes)")
    _require(len(hosts) == len(definition.host_element_ids) and
             all(type(h) is CartesianQ4Host for h in hosts), "exact declared host local inventory required")
    _require(len({n.node_id for n in nodes}) == count and len({n.q_offset for n in nodes}) == count,
             "duplicate/overlapping shell node mapping")
    _require(len({n.coordinates_m for n in nodes}) == count, "ambiguous coincident shell node identities")
    _require({h.element_id for h in hosts} == set(definition.host_element_ids), "declared host element mapping differs")
    _require(len({h.element_id for h in hosts}) == len(hosts), "duplicate host element identity")
    _require(beam_port.node_id == definition.beam_node_id and
             beam_port.q_offset not in {n.q_offset for n in nodes}, "physical beam node/coordinate mapping differs or overlaps")
    by_id = {n.node_id: n for n in nodes}
    _require(set(by_id) == set().union(*(set(h.node_ids) for h in hosts)),
             "complete and only selected host vertices required")
    geometry = [_host(h, by_id) for h in hosts]
    _require(all(g[2] == geometry[0][2] for g in geometry), "common flat XY shell reference plane required")
    _rectangular_hosts(geometry)
    bounds = tuple(map(F, definition.bounds_xy_m))
    area = (bounds[1]-bounds[0])*(bounds[3]-bounds[2])
    ordered_nodes = tuple(sorted(nodes, key=lambda n: n.q_offset))
    index = {n.node_id: i for i, n in enumerate(ordered_nodes)}
    mean = [F(0)]*count
    gradients = [[F(0)]*count for _ in range(2)]
    covered = F(0)
    for ids, host_bounds, _ in geometry:
        x0, x1, y0, y1 = host_bounds
        left, right = max(x0, bounds[0]), min(x1, bounds[1])
        low, high = max(y0, bounds[2]), min(y1, bounds[3])
        _require(right > left and high > low, "footprint must cross every selected host with positive area")
        ix, dx = _one_dimensional(left, right, x0, x1)
        iy, dy = _one_dimensional(low, high, y0, y1)
        covered += (right-left)*(high-low)
        for node, xside, yside in zip(ids, (0, 1, 1, 0), (0, 0, 1, 1)):
            i = index[node]
            mean[i] += ix[xside]*iy[yside]/area
            gradients[0][i] += dx[xside]*iy[yside]/area
            gradients[1][i] += ix[xside]*dy[yside]/area
    _require(covered == area, "incomplete or overlapping footprint host coverage")
    _require(sum(mean) == 1 and all(sum(row) == 0 for row in gradients), "exact reference partition/gradient failure")
    positions = [n.coordinates_m for n in ordered_nodes]
    for axis in range(2):
        for coordinate in range(2):
            _require(sum(gradients[axis][i]*F(positions[i][coordinate]) for i in range(count)) == int(axis == coordinate),
                     "exact Cartesian linear-field reproduction failure")
    centroid = ((bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, geometry[0][2])
    eccentric = tuple(F(value)-center for value, center in zip(beam_port.coordinates_m, centroid))
    try:
        m, g, center, e = np.asarray(mean, dtype=float), np.asarray(gradients, dtype=float), np.asarray(centroid, dtype=float), np.asarray(eccentric, dtype=float)
        area_value = float(area)
    except (OverflowError, ValueError) as error:
        raise ValueError("representable bounded reference geometry required") from error
    _require(isfinite(area_value) and area_value > 0. and all(np.isfinite(v).all() for v in (m, g, center, e)),
             "representable bounded reference geometry required")
    _require(all(value == 0 or converted != 0 for value, converted in zip(mean, m)) and
             all(value == 0 or converted != 0 for row, converted_row in zip(gradients, g) for value, converted in zip(row, converted_row)),
             "reference weight/gradient underflow is unsupported")
    trace = np.zeros((6, 6*count))
    for axis in range(3):
        trace[axis, axis::6] = m
    trace[3, 3::6], trace[4, 4::6] = m, m
    trace[5, 0::6], trace[5, 1::6] = -.5*g[1], .5*g[0]
    x, y, z = e
    cross = np.array(((0., -z, y), (z, 0., -x), (-y, x, 0.)))
    shell_map = -np.eye(6)
    shell_map[:3, 3:] = cross
    constraint = np.column_stack((shell_map@trace, np.eye(6)))
    _require(np.isfinite(trace).all() and np.isfinite(constraint).all(), "representable local distributing work map required")
    slots = [n.q_offset+d for n in ordered_nodes for d in range(6)] + [beam_port.q_offset+d for d in range(6)]
    binding = dict(law=LAW, footprint_id=definition.footprint_id, bounds=definition.bounds_xy_m,
                   hosts=sorted((h.element_id, h.node_ids) for h in hosts),
                   nodes=[(n.node_id, n.coordinates_m, n.q_offset) for n in ordered_nodes],
                   beam=(beam_port.node_id, beam_port.coordinates_m, beam_port.q_offset, beam_port.reference_axes),
                   mean=m.tolist(), gradients=g.tolist())
    identity = sha256(json.dumps(binding, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return CompiledDistributingFootprint(definition, tuple(n.node_id for n in ordered_nodes),
        tuple(sorted(definition.host_element_ids)), beam_port.node_id, np.asarray(slots), positions,
        beam_port.coordinates_m, center, e, area_value, m, g, trace, constraint, identity)
