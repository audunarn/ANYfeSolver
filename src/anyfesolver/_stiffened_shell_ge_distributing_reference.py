"""Private GE binding of the distributing reference-linear derivative/work map.

The standalone binder does not replace an owner's operator or publish state.
The same compiler also supplies explicitly admitted reference-linear owner rows.
Common reference-spatial covectors are not arbitrary raw Log coordinate moments.
No finite operator or mechanical accuracy qualification is provided.
"""
from dataclasses import dataclass, field
from hashlib import sha256
import json
import weakref

import numpy as np
from scipy import sparse

from ._stiffened_shell_distributing_footprint import (
    CartesianQ4Host, CompiledDistributingFootprint, DistributingFootprintDefinition,
    FlatQ4Node, IDENTITY_AXES, LAW, PhysicalBeamNodePort, compile_footprint,
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _owned(values):
    array = np.asarray(values)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _reference_owner(owner, *, allow_finite_material=False):
    from ._stiffened_shell_ge_experimental import ExperimentalGEAnalysis, Q4_ALGORITHMIC_PULLBACK_V1
    _require(type(owner) is ExperimentalGEAnalysis, "actual experimental GE owner required")
    owner._require_open()
    definition = owner._definition
    # The correspondence compiler is geometric. Its reference-linear public
    # binding remains elastic; only the selected finite material owner may reuse
    # the same immutable geometry through its private compiled attachment seam.
    finite_material = (allow_finite_material and owner._finite_distributing and
                       owner._distributing_material)
    _require(not owner._modal_only and not owner._curved_reference and
             (not owner._has_physical_fibre or
              (finite_material and owner._distributing_native)) and not owner._bl5_material and
             (definition.shell.j2_curve is None or finite_material) and not definition.shell.s3_elements and
             definition.q4_pullback_policy == Q4_ALGORITHMIC_PULLBACK_V1 and
             not definition.lines and definition.shell_reference_policy == "flat" and
             definition.bl5_material_policy == "elastic" and definition.modal_boundary_policy == "supported",
             "supported flat elastic full-map Q4/generalized native reference owner required")
    _require(owner._finite_driver is None and owner._accepted is None and
             owner._generation == 0 and not owner._writing,
             "unadvanced reference owner required; finite/state operators are excluded")


@dataclass(frozen=True, eq=False)
class ReferenceTransfer:
    """Resisting side covectors; externally exerted actions have opposite sign."""
    model_identity: str
    candidate_identity: str
    shell_node_ids: tuple[int, ...]
    beam_node_id: int
    shell_covectors: np.ndarray
    beam_covector: np.ndarray
    shell_force: np.ndarray
    shell_moment_about_centroid: np.ndarray
    centroid_m: np.ndarray
    eccentricity_m: np.ndarray

    def __post_init__(self):
        for name in ("shell_covectors", "beam_covector", "shell_force",
                     "shell_moment_about_centroid", "centroid_m", "eccentricity_m"):
            object.__setattr__(self, name, _owned(getattr(self, name)))


@dataclass(frozen=True, eq=False)
class BoundDistributingReference:
    """An immutable local map bound to one unadvanced open owner instance."""
    compiled: CompiledDistributingFootprint
    model_identity: str
    identity: str
    global_dimension: int
    generation: int
    constraint_scale: np.ndarray
    _owner_ref: object
    _K_ref: object
    _shell_parts_ref: object
    _data: np.ndarray
    _indices: np.ndarray
    _indptr: np.ndarray

    def __post_init__(self):
        for name in ("constraint_scale", "_data", "_indices", "_indptr"):
            object.__setattr__(self, name, _owned(getattr(self, name)))

    def _check(self, owner):
        from ._stiffened_shell_ge_experimental import ExperimentalGEAnalysis
        _require(type(owner) is ExperimentalGEAnalysis and self._owner_ref() is owner,
                 "distributing map belongs to another GE owner")
        owner._require_open()
        _require(owner._identity == self.model_identity and
                 owner._K.shape == (self.global_dimension, self.global_dimension) and
                 owner._generation == self.generation and owner._finite_driver is None and
                 owner._accepted is None and not owner._writing and owner._K is self._K_ref and
                 owner._shell_parts is self._shell_parts_ref,
                 "stale distributing reference model/state binding")
        _reference_owner(owner)
        shell_count = len(self.compiled.shell_node_ids)
        for node, slots, position in zip(self.compiled.shell_node_ids,
                self.compiled.gather_slots[:6*shell_count].reshape(shell_count, 6), self.compiled.shell_positions_m):
            index = owner._shell_index.get(node)
            _require(index is not None and 6*index == int(slots[0]) and
                     np.array_equal(owner._shell_xyz[index], position), "stale local shell mapping")
        beam = self.compiled.beam_node_id
        index = owner._beam_index.get(beam)
        _require(index is not None and owner._shell_count+6*index == int(self.compiled.gather_slots[-6]) and
                 beam in owner._beam_nodes and
                 np.array_equal(owner._beam_nodes[beam][0], self.compiled.beam_position_m) and
                 np.array_equal(owner._beam_nodes[beam][1], np.eye(3)), "stale physical beam mapping")

    def constraint_matrix(self, owner):
        """Own a canonical 6-by-fullN CSR scatter; never a global dense map."""
        self._check(owner)
        matrix = sparse.csr_matrix((self._data.copy(), self._indices.copy(), self._indptr.copy()),
                                   shape=(6, self.global_dimension))
        for name in ("data", "indices", "indptr"):
            setattr(matrix, name, _owned(getattr(matrix, name)))
        return matrix

    def gap(self, owner, physical_coordinates):
        self._check(owner)
        values = np.asarray(physical_coordinates)
        _require(values.shape == (self.global_dimension,), "exact full GE coordinate dimension required")
        _require(values.dtype.kind in "fiu" and np.isfinite(values).all(),
                 "complete finite real full GE coordinates required")
        return self.compiled.gap(self.compiled.gather(values))

    def side_actions(self, owner, multipliers):
        self._check(owner)
        local = self.compiled.transpose_work(multipliers).reshape(-1, 6)
        shell = local[:-1]
        with np.errstate(over="ignore", invalid="ignore"):
            force = shell[:, :3].sum(axis=0)
            moment = (np.cross(self.compiled.shell_positions_m-self.compiled.centroid_m,
                               shell[:, :3])+shell[:, 3:]).sum(axis=0)
        _require(np.isfinite(force).all() and np.isfinite(moment).all(),
                 "representable common reference-spatial side wrench required")
        return ReferenceTransfer(self.model_identity, self.identity, self.compiled.shell_node_ids,
            self.compiled.beam_node_id, shell, local[-1], force, moment,
            self.compiled.centroid_m, self.compiled.eccentricity_m)


def _compile_reference_geometry(definition, *, shell_parts, shell_index, shell_xyz,
                                beam_index, beam_nodes, shell_count, global_dimension):
    """Compile prepared reference maps without consulting owner state or admission."""
    _require(type(definition) is DistributingFootprintDefinition, "typed distributing definition required")
    wanted = set(definition.host_element_ids)
    hosts = []
    for eid, element, _slots, _origin in shell_parts:
        if eid in wanted:
            hosts.append(CartesianQ4Host(eid, tuple(element.node_ids)))
            if len(hosts) == len(wanted):
                break
    _require(len(hosts) == len(wanted), "selected actual GE shell element IDs required")
    selected_ids = set().union(*(set(h.node_ids) for h in hosts))
    nodes = tuple(FlatQ4Node(node, tuple(shell_xyz[shell_index[node]]),
                            6*shell_index[node]) for node in sorted(selected_ids))
    _require(definition.beam_node_id in beam_index, "existing physical native beam node required")
    position, axes = beam_nodes[definition.beam_node_id]
    _require(np.array_equal(axes, np.eye(3)), "identity physical beam reference triad required")
    beam = PhysicalBeamNodePort(definition.beam_node_id, tuple(position),
        shell_count+6*beam_index[definition.beam_node_id], IDENTITY_AXES)
    compiled = compile_footprint(definition, nodes, tuple(hosts), beam)
    _require(np.max(compiled.gather_slots) < global_dimension,
             "complete actual GE global coordinate mapping required")
    return compiled


def _scatter_reference(compiled, count):
    rows, columns = np.nonzero(compiled.constraint_matrix)
    matrix = sparse.coo_matrix((compiled.constraint_matrix[rows, columns],
                               (rows, compiled.gather_slots[columns])), shape=(6, count)).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    _require(matrix.has_canonical_format and np.isfinite(matrix.data).all(),
             "finite canonical sparse reference scatter required")
    return matrix


def bind_reference(owner, definition):
    """Select bounded actual shell cells once and bind the reference compiler."""
    from ._stiffened_shell_ge_experimental import _require_independent_supported_rows
    _reference_owner(owner)
    count = owner._K.shape[0]
    _require(owner._K.shape == (count, count), "square actual GE operator required")
    compiled = _compile_reference_geometry(definition, shell_parts=owner._shell_parts,
        shell_index=owner._shell_index, shell_xyz=owner._shell_xyz, beam_index=owner._beam_index,
        beam_nodes=owner._beam_nodes, shell_count=owner._shell_count, global_dimension=count)
    matrix = _scatter_reference(compiled, count)
    row_scale = np.asarray(np.abs(matrix)@owner._coordinate_scale).ravel()
    _require(np.isfinite(row_scale).all() and np.all(row_scale > 0.), "finite positive candidate row scales required")
    # All other columns are identically zero. Audit the identical supported row
    # rank on its bounded local incidence, avoiding a global dense array.
    local_free = np.flatnonzero(np.isin(compiled.gather_slots, owner._free))
    _require_independent_supported_rows(sparse.csr_matrix(compiled.constraint_matrix), local_free,
        owner._coordinate_scale[compiled.gather_slots], row_scale,
        tuple(f"footprint={definition.footprint_id} distributing reference row={i}" for i in range(6)))
    identity = sha256(json.dumps(dict(model=owner._identity, law=LAW, compiled=compiled.identity,
        hosts=compiled.host_element_ids, gather=compiled.gather_slots.tolist(),
        global_dimension=count, q4_policy=_definition_policy(owner),
        rows="common identity reference-spatial u/omega"), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return BoundDistributingReference(compiled, owner._identity, identity, count, owner._generation,
        row_scale, weakref.ref(owner), owner._K, owner._shell_parts,
        matrix.data, matrix.indices, matrix.indptr)


def _definition_policy(owner):
    return (owner._definition.q4_pullback_policy, owner._definition.shell_reference_policy,
            owner._definition.bl5_material_policy, owner._definition.modal_boundary_policy)


@dataclass(frozen=True)
class DistributingFootprintSelection:
    footprint_ids: tuple[str, ...]

    def __post_init__(self):
        try:
            ids = tuple(self.footprint_ids)
        except TypeError as error:
            raise ValueError("explicit distributing footprint IDs required") from error
        _require(ids and all(type(v) is str and v and v == v.strip() for v in ids) and
                 len(set(ids)) == len(ids), "distinct explicit distributing footprint IDs required")
        object.__setattr__(self, "footprint_ids", ids)


@dataclass(frozen=True, eq=False)
class DistributingFootprintWrench:
    """Physical exerted actions (-C.T lambda) in common reference-spatial rows.

    Shell nodal moments use original nodal origins; the shell resultant is about
    the area centroid, and the beam moment is about its actual nodal origin.
    Yaw is transferred through planar force couples, with no shell rz action.
    """
    footprint_id: str
    model_identity: str
    compiled_identity: str
    host_element_ids: tuple[int, ...]
    shell_node_ids: tuple[int, ...]
    beam_node_id: int
    centroid_m: np.ndarray
    beam_position_m: np.ndarray
    eccentricity_m: np.ndarray
    area_m2: float
    shell_node_actions: np.ndarray
    shell_action: np.ndarray
    beam_action: np.ndarray
    gap: np.ndarray
    generation: int
    configuration: str = field(default="reference_linear", init=False)
    law: str = field(default=LAW, init=False)
    production_qualified: bool = field(default=False, init=False)

    def __post_init__(self):
        for name, shape in (("centroid_m", (3,)), ("beam_position_m", (3,)),
                            ("eccentricity_m", (3,)), ("shell_node_actions", (len(self.shell_node_ids), 6)),
                            ("shell_action", (6,)), ("beam_action", (6,)), ("gap", (6,))):
            values = np.asarray(getattr(self, name))
            _require(values.shape == shape and values.dtype.kind in "fiu" and np.isfinite(values).all(),
                     "finite real distributing recovery arrays required")
            object.__setattr__(self, name, _owned(values))
        object.__setattr__(self, "host_element_ids", tuple(self.host_element_ids))
        object.__setattr__(self, "shell_node_ids", tuple(self.shell_node_ids))


def _reference_records(owner, q, multipliers, generation):
    records = []
    for compiled, start in owner._distributing_bindings:
        local = -compiled.transpose_work(multipliers[start:start+6]).reshape(-1, 6)
        shell = local[:-1]
        with np.errstate(over="ignore", invalid="ignore"):
            force = shell[:, :3].sum(axis=0)
            moment = (np.cross(compiled.shell_positions_m-compiled.centroid_m,
                               shell[:, :3])+shell[:, 3:]).sum(axis=0)
        records.append(DistributingFootprintWrench(compiled.definition.footprint_id, owner._identity,
            compiled.identity, compiled.host_element_ids, compiled.shell_node_ids, compiled.beam_node_id,
            compiled.centroid_m, compiled.beam_position_m, compiled.eccentricity_m, compiled.area_m2,
            shell, np.r_[force, moment], local[-1], compiled.gap(compiled.gather(q)), generation))
    return tuple(records)


def _selected_records(result, selection):
    _require(type(selection) is DistributingFootprintSelection, "typed distributing footprint selection required")
    records = {record.footprint_id: record for record in result.distributing_footprint_wrenches}
    _require(all(name in records for name in selection.footprint_ids), "unknown distributing footprint recovery ID")
    return tuple(records[name] for name in selection.footprint_ids)
