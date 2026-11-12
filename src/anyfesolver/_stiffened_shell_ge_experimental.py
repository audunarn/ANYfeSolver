"""Experimental sparse reference-linear GE/Q4 point-connected panel.

This is an integration slice, not an accepted G0/G1 formulation.  It retains
all native B3 cell variables and uses the existing objective point operator.
Qualified line coupling and material updates are separate work. The optional
L1 weak-line route is reference-linear and explicitly unqualified.
The BL5 finite route admits elastic response only and remains unqualified.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from copy import copy
from hashlib import sha256
from functools import wraps
from threading import RLock
import json
import warnings

from anymaterial import DNVC208MaterialCurve
import numpy as np
from scipy import linalg, sparse
from scipy.sparse import linalg as spla
from scipy.sparse.csgraph import maximum_bipartite_matching

from ._ge_beam3_native_definition import NativeBeamDefinition
from ._ge_beam3_fibre_section import PhysicalFibreSection
from ._ge_beam3_pose_joint import RigidPoseJoint
from ._stiffened_shell_point_recovery import PointTransferWrench, reference_point_records
from ._stiffened_shell_ge_distributing_finite import (
    FiniteDistributingFootprintDefinition, PreparedFiniteDistributingAttachment,
)
from ._stiffened_shell_distributing_pose import FiniteDistributingPose, LAW as DISTRIBUTING_FINITE_LAW
from ._stiffened_shell_ge_distributing_reference import (
    DistributingFootprintDefinition, DistributingFootprintSelection, DistributingFootprintWrench,
    LAW as DISTRIBUTING_REFERENCE_LAW, _compile_reference_geometry, _scatter_reference,
    _reference_records as distributing_reference_records, _selected_records as selected_distributing_records,
)
from ._stiffened_shell_nodal_footprint import (
    NodalRigidFootprint, NodalRigidFootprintSelection, NodalRigidFootprintWrench,
    validate_footprints, selected_wrenches, stable_s3_policy_descriptor,
    retained_scale_policy_descriptor, chart_admission_policy_descriptor,
    require_rotation_triples,
    CHART_ADMISSION_POLICY, INCREMENT_ORIGIN_CHART_POLICY,
)
from ._ge_beam3_generalized_static_boundary import spatial_jacobian
from ._stiffened_shell_s3_algorithmic_pullback import POLICY as S3_J2_PULLBACK_POLICY
from ._stiffened_shell_q4_algorithmic_pullback import (
    POLICY as Q4_ALGORITHMIC_PULLBACK_V1, HISTORICAL_POLICY as Q4_HISTORICAL_POLICY,
    response as q4_algorithmic_response,
)
from .corotational import corotational_element_response
from ._stiffened_shell_coordinate_support import (
    POLICY as NATIVE_SHELL_COORDINATE_SUPPORT_V1,
    LEGACY as HISTORICAL_SHELL_SUPPORT_POLICY,
)
from ._stiffened_shell_q4_reference import elastic_reference_response
from .elements import create_shell_element
from .fe_core import FEModel
from .control import cancellation_safe_point
from ._stiffened_shell_ge_checkpoint_binary import (
    MAGIC as BINARY_CHECKPOINT_MAGIC, decode as _decode_binary_checkpoint,
    encode as _encode_binary_checkpoint, S3_MAGIC as S3_BINARY_CHECKPOINT_MAGIC,
    S3_SCHEMA as FINITE_S3_MATERIAL_CHECKPOINT,
    decode_s3 as _decode_s3_binary_checkpoint,
    CHART_MAGIC as CHART_CHECKPOINT_MAGIC, decode_chart as _decode_chart_checkpoint,
    NATIVE_SHELL_MAGIC, NATIVE_SHELL_SCHEMA, STABLE_NATIVE_SCHEMA,
    SCHEMAS as _BINARY_SCHEMAS, CURVED_POINT_MATERIAL_KINDS,
    FINITE_DISTRIBUTING_MATERIAL_KINDS,
    FINITE_DISTRIBUTING_NATIVE_KINDS,
)
from ._stiffened_shell_increment_control import (
    MAGIC as INCREMENT_CONTROL_MAGIC, decode_control, validate_control,
)
from ._stiffened_shell_l1_assembly_experimental import (
    ExperimentalL1Line, ExperimentalBL5Line,
    compile_lines as _compile_weak_lines,
    _row_count as _weak_line_row_count,
)


PROFILE = "B3_GE_E4_PL_STIFFENED_SHELL_V1"
BL5_GENERALIZED_MATERIAL_V1 = "BL5_GENERALIZED_MATERIAL_V1_DEV"
BL5_COUPLED_J2_MATERIAL_V1 = "BL5_COUPLED_J2_MATERIAL_V1_DEV"
BL5_ELASTIC_NATIVE_FIBRE_V1 = "BL5_ELASTIC_NATIVE_FIBRE_V1_DEV"
BL5_ELASTIC_NATIVE_FIBRE_MIXED_SHELL_V1 = "BL5_ELASTIC_NATIVE_FIBRE_MIXED_SHELL_V1_DEV"
BL5_NATIVE_FIBRE_MATERIAL_V1 = "BL5_NATIVE_FIBRE_MATERIAL_V1_DEV"
BL5_NATIVE_FIBRE_MIXED_SHELL_MATERIAL_V1 = "BL5_NATIVE_FIBRE_MIXED_SHELL_MATERIAL_V1_DEV"
GE_FINITE_DISTRIBUTING_GENERALIZED_MATERIAL_V1 = "GE_FINITE_DISTRIBUTING_GENERALIZED_MATERIAL_V1_DEV"
GE_FINITE_DISTRIBUTING_COUPLED_J2_MATERIAL_V1 = "GE_FINITE_DISTRIBUTING_COUPLED_J2_MATERIAL_V1_DEV"
GE_FINITE_DISTRIBUTING_GENERALIZED_CYCLIC_MATERIAL_V1 = "GE_FINITE_DISTRIBUTING_GENERALIZED_CYCLIC_MATERIAL_V1_DEV"
GE_FINITE_DISTRIBUTING_COUPLED_J2_CYCLIC_MATERIAL_V1 = "GE_FINITE_DISTRIBUTING_COUPLED_J2_CYCLIC_MATERIAL_V1_DEV"
GE_FINITE_DISTRIBUTING_NATIVE_CYCLIC_MATERIAL_V1 = "GE_FINITE_DISTRIBUTING_NATIVE_CYCLIC_MATERIAL_V1_DEV"
GE_FINITE_DISTRIBUTING_NATIVE_COUPLED_J2_CYCLIC_MATERIAL_V1 = "GE_FINITE_DISTRIBUTING_NATIVE_COUPLED_J2_CYCLIC_MATERIAL_V1_DEV"
CHECKPOINT = "B3_GE_E4_PL_STIFFENED_SHELL_EXPERIMENTAL_LINEAR_V1"
FINITE_CHECKPOINT = "B3_GE_E4_PL_STIFFENED_SHELL_EXPERIMENTAL_FINITE_V1"
FINITE_MATERIAL_CHECKPOINT = (
    "B3_GE_E4_PL_STIFFENED_SHELL_EXPERIMENTAL_FINITE_MATERIAL_V3")
FINITE_SHELL_MATERIAL_CHECKPOINT = (
    "B3_GE_E4_PL_STIFFENED_SHELL_EXPERIMENTAL_FINITE_SHELL_J2_V4")
MAX_JSON_CHECKPOINT_BYTES = 2_000_000
MAX_JSON_CHECKPOINT_VALUES = 50_000
_MAX_DENSE_CONSTRAINT_ROWS = 48
_MAX_DENSE_CONSTRAINT_VALUES = 16_384


def _state_writer(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._operation_lock:
            if self._writing:
                raise RuntimeError("reentrant mixed state mutation is not permitted")
            self._writing = True
            try:
                return method(self, *args, **kwargs)
            finally:
                self._writing = False
    return call


def _accepted_snapshot(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._commit_lock:
            return method(self, *args, **kwargs)
    return call


def _require_independent_supported_rows(constraints, free, coordinate_scale,
                                        row_scale, sources=()):
    """Check small numerical ranks; keep large structural checks sparse.

    Structural rank cannot certify numerical independence. The subsequent
    sparse mixed factorization remains the numerical check for large models.
    """
    reduced = constraints[:, free].tocsr()
    rows, columns = reduced.shape
    if sources and len(sources) != rows:
        raise ValueError("complete attachment row provenance required")

    def failure(message, row):
        suffix = f": {sources[row]}" if sources else ""
        raise ValueError(message+suffix)

    if rows > columns:
        failure("independent supported attachment rows required", columns)
    if rows <= _MAX_DENSE_CONSTRAINT_ROWS and rows*columns <= _MAX_DENSE_CONSTRAINT_VALUES:
        scaled = (sparse.diags(1./row_scale) @ reduced @
                  sparse.diags(coordinate_scale[free])).toarray()
        rank = np.linalg.matrix_rank(scaled)
        if rank != rows:
            _, _, pivot = linalg.qr(scaled.T, pivoting=True, mode="economic")
            failure("independent supported attachment rows required",
                    int(pivot[rank]))
        return
    pattern = reduced.copy()
    pattern.eliminate_zeros()
    matching = maximum_bipartite_matching(pattern, perm_type="column")
    if np.any(matching < 0):
        failure("structurally independent supported attachment rows required",
                int(np.flatnonzero(matching < 0)[0]))


@dataclass(frozen=True)
class ShellPanel:
    node_ids: tuple[int, ...]
    coordinates_m: tuple[tuple[float, float, float], ...]
    q4_elements: tuple[tuple[int, int, int, int], ...]
    thickness_m: float
    young_pa: float
    poisson: float
    density_kg_m3: float
    s3_elements: tuple[tuple[int, int, int], ...] = ()
    j2_curve: DNVC208MaterialCurve | None = None


@dataclass(frozen=True)
class PointConnection:
    shell_node: int
    beam_node: int


@dataclass(frozen=True)
class FixedCoordinate:
    domain: str
    node: int
    component: int


@dataclass(frozen=True)
class NodalAction:
    domain: str
    node: int
    wrench: tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class LinearDefinition:
    profile: str
    shell: ShellPanel
    beams: tuple[NativeBeamDefinition, ...]
    points: tuple[PointConnection, ...]
    fixed: tuple[FixedCoordinate, ...]
    lines: tuple[ExperimentalL1Line | ExperimentalBL5Line, ...] = ()
    q4_pullback_policy: str = Q4_HISTORICAL_POLICY
    footprints: tuple[NodalRigidFootprint, ...] = ()
    modal_boundary_policy: str = "supported"
    footprint_chart_policy: str = CHART_ADMISSION_POLICY
    bl5_material_policy: str = "elastic"
    shell_reference_policy: str = "flat"
    retained_scale_policy: str = "profile"
    distributing_footprints: tuple[DistributingFootprintDefinition, ...] = ()
    finite_distributing_footprints: tuple[FiniteDistributingFootprintDefinition, ...] = ()
    finite_distributing_material_policy: str = "elastic"
    cyclic_reference_actions: tuple[NodalAction, ...] = ()
    shell_support_policy: str = HISTORICAL_SHELL_SUPPORT_POLICY


@dataclass(frozen=True)
class LinearResult:
    """Experimental linear result with dimensionless work/row-scaled errors."""
    status: str
    shell_displacement: np.ndarray
    beam_coordinates: np.ndarray
    multipliers: np.ndarray
    reactions: np.ndarray
    free_equilibrium_ratio: float
    constraint_max: float
    generation: int
    production_qualified: bool = False
    point_wrenches: tuple[PointTransferWrench, ...] = ()
    body_load: object | None = None
    distributed_load: object | None = None
    pressure_load: object | None = None
    distributing_footprint_wrenches: tuple[DistributingFootprintWrench, ...] = ()


def _owned(values, shape=None):
    result = np.array(values, dtype=float, copy=True)
    if (shape is not None and result.shape != shape) or not np.isfinite(result).all():
        raise ValueError("finite owned stiffened-shell values required")
    return np.frombuffer(result.tobytes(), dtype=float).reshape(result.shape)


CURVED_Q4_POINT_REFERENCE_V1 = "CURVED_Q4_POINT_REFERENCE_V1_DEV"
CURVED_Q4_POINT_MATERIAL_REFERENCE_V1 = "CURVED_Q4_POINT_MATERIAL_REFERENCE_V1_DEV"
CURVED_Q4_S3_POINT_REFERENCE_V1 = "CURVED_Q4_S3_POINT_REFERENCE_V1_DEV"


def _shell_reference(panel: ShellPanel, *, curved=False, oriented=False):
    ids = tuple(panel.node_ids)
    xyz = _owned(panel.coordinates_m, (len(ids), 3))
    if (len(ids) < 3 or len(ids) != len(set(ids)) or
            len(panel.q4_elements)+len(panel.s3_elements) < 1 or
            any(type(v) is not int or v <= 0 for v in ids) or
            (not curved and not np.allclose(xyz[:, 2], xyz[0, 2], rtol=0., atol=1e-12)) or
            not (np.isfinite(panel.thickness_m) and panel.thickness_m > 0. and
                 np.isfinite(panel.young_pa) and panel.young_pa > 0. and
                 np.isfinite(panel.poisson) and -1. < panel.poisson < .5 and
                 np.isfinite(panel.density_kg_m3) and panel.density_kg_m3 > 0.)):
        raise ValueError("experimental connected flat Q4/S3 panel required")
    if panel.j2_curve is not None:
        if (type(panel.j2_curve) is not DNVC208MaterialCurve or
                any(not np.isfinite(value)
                    for value in panel.j2_curve.as_dict().values())):
            raise ValueError("experimental shell J2 requires finite DNVC-C208 data")
    index = {node: i for i, node in enumerate(ids)}
    model = FEModel("experimental-stiffened-shell-q4")
    model.add_material("shell", panel.young_pa, panel.poisson,
                       density=panel.density_kg_m3,
                       yield_stress=(panel.j2_curve.sigma_prop
                                     if panel.j2_curve is not None else 0.),
                       hardening_curve=panel.j2_curve)
    for node, point in zip(ids, xyz):
        model.add_node(node, *map(float, point))
    pieces = []
    cells = (tuple((nodes, "e4-pl") for nodes in panel.q4_elements) +
             tuple((nodes, "e4-pl-s3-v2d") for nodes in panel.s3_elements))
    neighbours = {node: set() for node in ids}
    cell_keys = set()
    oriented_edges = {}
    for eid, (nodes, formulation) in enumerate(cells, 1):
        if (len(nodes) != (4 if formulation == "e4-pl" else 3) or
                len(set(nodes)) != len(nodes) or
                any(node not in index for node in nodes)):
            raise ValueError("distinct known Q4/S3 nodes required")
        key = (formulation, frozenset(nodes))
        if key in cell_keys:
            raise ValueError("duplicate experimental shell element")
        cell_keys.add(key)
        if oriented:
            loop = tuple(nodes)
            for a, b in zip(loop, loop[1:]+loop[:1]):
                edge = (min(a, b), max(a, b))
                previous = oriented_edges.get(edge)
                if previous is None and edge in oriented_edges:
                    raise ValueError("nonmanifold curved Q4/S3 edge")
                if previous == (a, b):
                    raise ValueError("inconsistent oriented curved Q4/S3 adjacent facets")
                oriented_edges[edge] = None if previous is not None else (a, b)
        for node in nodes:
            neighbours[node].update(other for other in nodes if other != node)
        normal = np.array((0.,0.,1.))
        if curved:
            vertices=xyz[[index[node] for node in nodes]]
            # The Q4 perimeter closes through vertex 3; S3 closes through 2.
            # Preserve each element's declared orientation in the reference.
            side=3 if formulation=="e4-pl" else 2
            normal=np.cross(vertices[1]-vertices[0],vertices[side]-vertices[0])
            norm=float(np.linalg.norm(normal));length=float(np.max(np.linalg.norm(vertices-vertices[0],axis=1)))
            if not np.isfinite(norm) or norm<=1e-12*length*length:
                raise ValueError("nondegenerate curved Q4/S3 facet required")
            normal/=norm
            if formulation=="e4-pl" and abs(float((vertices[2]-vertices[0])@normal))>1e-10*length:
                raise ValueError("planar curved-reference Q4 facets required")
        element = create_shell_element(
            eid, list(nodes), "shell", formulation=formulation,
            thickness=panel.thickness_m, reference_normal=tuple(normal))
        model.add_element(eid, element)
        slots = np.array([6*index[node]+axis for node in nodes for axis in range(6)])
        pieces.append((eid, element, slots, formulation))
    seen = {ids[0]}
    pending = [ids[0]]
    while pending:
        for other in neighbours[pending.pop()]-seen:
            seen.add(other)
            pending.append(other)
    if len(seen) != len(ids):
        raise ValueError("connected Q4/S3 shell cells must use every node")
    pieces = [
        (eid, element, slots,
         element.init_model_bound_nonlinear_state(
             model.mesh, model.get_material("shell"), 3)
         if formulation == "e4-pl-s3-v2d" else element.init_nonlinear_state(3))
        for eid, element, slots, formulation in pieces
    ]
    return model, ids, xyz, index, pieces


def _beam_reference(definitions: tuple[NativeBeamDefinition, ...]):
    if len(definitions) < 1 or any(type(v) is not NativeBeamDefinition
                                   for v in definitions):
        raise ValueError("explicit native B3 definition required")
    elements = []
    inertias = []
    nodes = {}
    for definition in definitions:
        element, _inertia = definition.instantiate()
        inertias.append(_inertia)
        if (type(element.section).__name__ != "EllipsoidalGeneralizedSection" and
                type(element.section) is not PhysicalFibreSection):
            raise ValueError("explicit native generalized or physical-fibre section required")
        ref = element.operator.reference
        for node, position, frame in zip(element.node_ids, ref.coordinates,
                                         ref.nodal_triads):
            value = (_owned(position, (3,)), _owned(frame, (3, 3)))
            previous = nodes.setdefault(node, value)
            if (not np.array_equal(previous[0], value[0]) or
                    not np.array_equal(previous[1], value[1])):
                raise ValueError("shared B3 node has inconsistent position or frame")
        elements.append(element)
    ids = tuple(sorted(nodes))
    index = {node: i for i, node in enumerate(ids)}
    nodal = 6*len(ids)
    pieces = []
    for i, element in enumerate(elements):
        slots = np.array([6*index[node]+axis for node in element.node_ids
                          for axis in range(6)] + list(range(nodal+24*i,
                                                             nodal+24*(i+1))))
        pieces.append((element, slots, element.operator.cell.virgin()))
    return ids, index, nodes, pieces, nodal+24*len(pieces), tuple(inertias)


def _assemble_shell(model, pieces, count, coordinate_scale,
                    q4_pullback_policy=Q4_HISTORICAL_POLICY):
    rows, columns, data = [], [], []
    residual = np.zeros(count)
    local_error = 0.
    for eid, element, slots, origin in pieces:
        if len(slots) == 18:
            force, tangent, candidate = element.compute_nonlinear_response(
                model.mesh, model.get_material("shell"), np.zeros(18),
                origin, 3, True)
        elif q4_pullback_policy == Q4_ALGORITHMIC_PULLBACK_V1:
            force, tangent, candidate = q4_algorithmic_response(
                model, element, np.zeros(24), origin, 3)
        else:
            response = elastic_reference_response(model, eid, element, origin)
            if response is None:
                response = corotational_element_response(
                    model, eid, element, np.zeros(24), True,
                    committed_state=origin, num_layers=3,
                    tangent_mode="consistent")
            force, tangent, candidate = response
        if (force is None or tangent is None or candidate is None or
                not np.isfinite(force).all() or not np.isfinite(tangent).all()):
            raise RuntimeError("finite Q4 reference operator unavailable")
        local_error = max(local_error, _scaled_reference_residual(
            force, tangent, coordinate_scale[slots]))
        np.add.at(residual, slots, force)
        rows.extend(np.repeat(slots, len(slots)))
        columns.extend(np.tile(slots, len(slots)))
        data.extend(np.asarray(tangent).ravel())
    return (residual, sparse.coo_matrix((data, (rows, columns)),
                                        shape=(count, count)).tocsr(), local_error)


def _assemble_beam(pieces, nodes, index, count, coordinate_scale):
    rows, columns, data = [], [], []
    residual = np.zeros(count)
    local_error = 0.
    for element, slots, origin in pieces:
        ref = element.operator.reference
        position = np.array([nodes[node][0] for node in element.node_ids])
        frames = np.array([nodes[node][1] for node in element.node_ids])
        response = element.operator.evaluate(
            position, np.zeros((3, 3)), frames,
            np.tile(np.eye(3), (2, 1, 1)), np.zeros(18),
            origin=origin, check=element.operator.guard)
        local = spatial_jacobian(response.residual,
                                 response.hessian+response.hessian_low)
        if local.shape != (42, 42) or not np.isfinite(local).all():
            raise RuntimeError("finite native B3 reference operator unavailable")
        local_error = max(local_error, _scaled_reference_residual(
            response.residual, local, coordinate_scale[slots]))
        np.add.at(residual, slots, response.residual)
        rows.extend(np.repeat(slots, 42))
        columns.extend(np.tile(slots, 42))
        data.extend(local.ravel())
    return (residual, sparse.coo_matrix((data, (rows, columns)),
                                        shape=(count, count)).tocsr(), local_error)


def _reference_section_tangent(section):
    """Virgin elastic derivative for compilation scales, never a trial cache."""
    if type(section) is PhysicalFibreSection:
        response = section.response(np.zeros(6), section.virgin())
        return response.tangent + response.tangent_low
    return section.elastic


def _reference_scales(shell_xyz, beam_nodes, beam_parts, shell_count,
                      beam_count, *, footprint_retained_roles=False):
    """Freeze compilation scales; selected new profiles use native retained roles.

    Legacy owners retain their historical arrangement exactly. The footprint and
    coupled-J2 BL5 policies select native retained force/moment work pairing.
    """
    positions = np.vstack((shell_xyz, [value[0] for value in beam_nodes.values()]))
    centered = positions-np.mean(positions, axis=0)
    length = float(np.sqrt(np.mean(np.sum(centered*centered, axis=1))))
    if not np.isfinite(length) or length <= np.finfo(float).eps:
        raise ValueError("nondegenerate reference length required")
    pose_scale = np.array((length, length, length, 1., 1., 1.))
    shell_scale = np.tile(pose_scale, shell_count//6)
    nodal_count = beam_count-24*len(beam_parts)
    beam_scale = np.empty(beam_count)
    beam_scale[:nodal_count] = np.tile(pose_scale, nodal_count//6)
    strain_scale = np.array((1., 1., 1., 1./length, 1./length, 1./length))
    for i, (element, _slots, _origin) in enumerate(beam_parts):
        base = nodal_count+24*i
        beam_scale[base:base+6] = 1.
        resultant_scale = np.abs(_reference_section_tangent(element.section))@strain_scale
        # Native p consists of six independent force coordinates followed by
        # twelve moment coordinates, not three recovered station wrenches.
        # Preserve the historical arrangement for policies that have not selected
        # the native retained-role scaling contract.
        if footprint_retained_roles:
            beam_scale[base+6:base+24] = np.r_[np.tile(resultant_scale[:3], 2),
                                             np.tile(resultant_scale[3:], 4)]
        else:
            beam_scale[base+6:base+24] = np.tile(resultant_scale, 3)
    if not np.isfinite(shell_scale).all() or not np.isfinite(beam_scale).all():
        raise ValueError("finite reference coordinate scales required")
    return length, shell_scale, beam_scale


def _scaled_reference_residual(residual, tangent, coordinate_scale):
    """Compare each covector row with a same-unit tangent action."""
    row_scale = np.asarray(np.abs(tangent)@coordinate_scale).reshape(-1)
    if (not np.isfinite(residual).all() or not np.isfinite(row_scale).all() or
            np.any(row_scale < 0.)):
        raise ValueError("finite stress-free reference operator required")
    error = np.zeros_like(residual)
    np.divide(np.abs(residual), row_scale, out=error, where=row_scale > 0.)
    if np.any((row_scale == 0.) & (residual != 0.)):
        return float("inf")
    return float(np.max(error))


def _relative_equilibrium(residual, applied, coordinate_scale, floor_work):
    """Work-scaled mixed equilibrium ratio with a fixed physical work floor.

    Each covector row is paired with its frozen characteristic coordinate.
    This keeps forces, moments and native retained-variable equations in
    work units when supplied scales match the actual coordinate roles; unit
    invariance assumes that same correctly packed work-conjugate pairing.
    ``floor_work`` is one newton times the frozen reference length in SI.
    """
    residual = np.asarray(residual, dtype=float)
    applied = np.asarray(applied, dtype=float)
    coordinate_scale = np.asarray(coordinate_scale, dtype=float)
    if (residual.shape != applied.shape or residual.shape !=
            coordinate_scale.shape or residual.ndim != 1 or
            not np.isfinite(residual).all() or
            not np.isfinite(applied).all() or
            not np.isfinite(coordinate_scale).all() or
            np.any(coordinate_scale <= 0.) or
            not np.isfinite(floor_work) or floor_work <= 0.):
        raise ValueError("finite positive mixed work scales required")
    weighted_residual = residual*coordinate_scale
    weighted_applied = applied*coordinate_scale
    if (not np.isfinite(weighted_residual).all() or
            not np.isfinite(weighted_applied).all()):
        raise ValueError("representable mixed work residual required")
    scale = max(float(floor_work),
                float(np.max(np.abs(weighted_residual), initial=0.)),
                float(np.max(np.abs(weighted_applied), initial=0.)))
    return float(np.linalg.norm(weighted_residual/scale) /
                 max(floor_work/scale,
                     float(np.linalg.norm(weighted_applied/scale))))


def _linear_equilibrium(residual, applied, coordinate_scale):
    """Scale linear equilibrium to its actual load, including quiet load rows.

    A fixed work floor can accept zero response to a small applied load. The
    active-row check also prevents a large load from hiding a missed smaller
    load in a separate physical coordinate. A numerically unresolved mixed
    load fails closed; it is not declared an equilibrium state.
    """
    residual = np.asarray(residual, dtype=float)
    applied = np.asarray(applied, dtype=float)
    coordinate_scale = np.asarray(coordinate_scale, dtype=float)
    if (residual.ndim != 1 or residual.shape != applied.shape or
            residual.shape != coordinate_scale.shape or
            not np.isfinite(residual).all() or
            not np.isfinite(applied).all() or
            not np.isfinite(coordinate_scale).all() or
            np.any(coordinate_scale <= 0.)):
        raise ValueError("finite positive mixed work scales required")
    weighted_residual = residual*coordinate_scale
    weighted_applied = applied*coordinate_scale
    if (not np.isfinite(weighted_residual).all() or
            not np.isfinite(weighted_applied).all()):
        raise ValueError("representable mixed work residual required")
    magnitude = max(float(np.max(np.abs(weighted_residual), initial=0.)),
                    float(np.max(np.abs(weighted_applied), initial=0.)))
    if magnitude == 0.:
        return 0.
    applied_norm = float(np.linalg.norm(weighted_applied/magnitude))
    global_ratio = (float(np.linalg.norm(weighted_residual/magnitude)) /
                    applied_norm if applied_norm else float("inf"))
    active = applied != 0.
    with np.errstate(divide="ignore", over="ignore"):
        active_ratio = float(np.max(np.abs(residual[active]/applied[active]),
                                    initial=0.))
    return max(global_ratio, active_ratio)


def _scaled_constraint_gap(gap, row_scale):
    """Normalize each translation/rotation or weak-line row by fixed work map."""
    gap = np.asarray(gap, dtype=float)
    row_scale = np.asarray(row_scale, dtype=float)
    if (gap.ndim != 1 or gap.shape != row_scale.shape or
            not np.isfinite(gap).all() or
            not np.isfinite(row_scale).all() or
            np.any(row_scale <= 0.)):
        raise ValueError("finite positive constraint row scales required")
    return float(np.max(np.abs(gap)/row_scale, initial=0.))


class ExperimentalGEAnalysis:
    """Owned linear reference slice; no nonlinear or accepted gate claim."""

    def __init__(self, definition: LinearDefinition, resources=None):
        if (type(definition) is not LinearDefinition or
                definition.profile != PROFILE or resources is not None):
            raise ValueError("explicit experimental GE definition required")
        if (type(definition.shell_support_policy) is not str or
                definition.shell_support_policy not in (HISTORICAL_SHELL_SUPPORT_POLICY,NATIVE_SHELL_COORDINATE_SUPPORT_V1)):
            raise ValueError("explicit shell support policy required")
        self._coordinate_support = definition.shell_support_policy == NATIVE_SHELL_COORDINATE_SUPPORT_V1
        if self._coordinate_support and (type(definition.shell) is not ShellPanel or
                type(definition.fixed) is not tuple or any(type(s) is not FixedCoordinate or
                    type(s.component) is not int or s.domain not in ('shell','beam') or not 0<=s.component<6 for s in definition.fixed)):
            raise ValueError('coordinate shell supports require typed shell and fixed-coordinate records')
        if self._coordinate_support and (definition.lines or definition.footprints or
                definition.distributing_footprints or definition.finite_distributing_footprints or
                definition.shell.j2_curve is not None or definition.modal_boundary_policy != 'supported' or
                definition.bl5_material_policy != 'elastic' or definition.shell_reference_policy != 'flat' or
                definition.retained_scale_policy != 'profile' or definition.cyclic_reference_actions or
                (definition.shell.q4_elements and definition.q4_pullback_policy != Q4_ALGORITHMIC_PULLBACK_V1)):
            raise ValueError("coordinate shell supports require elastic native point Q4 algorithmic or S3 fixed-force scope")
        if self._coordinate_support:
            for node in {s.node for s in definition.fixed if s.domain == 'beam'}:
                rotational={s.component for s in definition.fixed if s.domain == 'beam' and s.node == node and s.component>=3}
                if rotational and rotational != {3,4,5}:
                    raise ValueError("coordinate shell supports exclude partial beam rotational supports")
        groups = definition.distributing_footprints
        if type(groups) is not tuple or any(type(g) is not DistributingFootprintDefinition for g in groups):
            raise ValueError("explicit typed distributing footprint tuple required")
        groups = tuple(DistributingFootprintDefinition(g.footprint_id, g.host_element_ids,
            g.beam_node_id, g.bounds_xy_m) for g in groups)
        self._distributing_reference = bool(groups)
        finite_groups = definition.finite_distributing_footprints
        if type(finite_groups) is not tuple or any(type(g) is not FiniteDistributingFootprintDefinition for g in finite_groups):
            raise ValueError("explicit typed finite distributing footprint tuple required")
        finite_groups = tuple(FiniteDistributingFootprintDefinition(g.footprint) for g in finite_groups)
        self._finite_distributing = bool(finite_groups)
        if (type(definition.finite_distributing_material_policy) is not str or
                definition.finite_distributing_material_policy not in (
                    "elastic", GE_FINITE_DISTRIBUTING_GENERALIZED_MATERIAL_V1,
                    GE_FINITE_DISTRIBUTING_COUPLED_J2_MATERIAL_V1,
                    GE_FINITE_DISTRIBUTING_GENERALIZED_CYCLIC_MATERIAL_V1,
                    GE_FINITE_DISTRIBUTING_COUPLED_J2_CYCLIC_MATERIAL_V1,
                    GE_FINITE_DISTRIBUTING_NATIVE_CYCLIC_MATERIAL_V1,
                    GE_FINITE_DISTRIBUTING_NATIVE_COUPLED_J2_CYCLIC_MATERIAL_V1)):
            raise ValueError("explicit admitted finite distributing material policy required")
        self._distributing_material = definition.finite_distributing_material_policy != "elastic"
        self._distributing_native = definition.finite_distributing_material_policy in (
            GE_FINITE_DISTRIBUTING_NATIVE_CYCLIC_MATERIAL_V1,
            GE_FINITE_DISTRIBUTING_NATIVE_COUPLED_J2_CYCLIC_MATERIAL_V1)
        self._distributing_shell_material = (
            definition.finite_distributing_material_policy in (
                GE_FINITE_DISTRIBUTING_COUPLED_J2_MATERIAL_V1,
                GE_FINITE_DISTRIBUTING_COUPLED_J2_CYCLIC_MATERIAL_V1,
                GE_FINITE_DISTRIBUTING_NATIVE_COUPLED_J2_CYCLIC_MATERIAL_V1))
        self._distributing_cyclic = definition.finite_distributing_material_policy in (
            GE_FINITE_DISTRIBUTING_GENERALIZED_CYCLIC_MATERIAL_V1,
            GE_FINITE_DISTRIBUTING_COUPLED_J2_CYCLIC_MATERIAL_V1,
            GE_FINITE_DISTRIBUTING_NATIVE_CYCLIC_MATERIAL_V1,
            GE_FINITE_DISTRIBUTING_NATIVE_COUPLED_J2_CYCLIC_MATERIAL_V1)
        if (type(definition.cyclic_reference_actions) is not tuple or
                (not self._distributing_cyclic and definition.cyclic_reference_actions)):
            raise ValueError("reference actions require the selected cyclic material history")
        if self._distributing_cyclic and not definition.cyclic_reference_actions:
            raise ValueError("cyclic history requires explicit reference actions")
        if self._distributing_material and not self._finite_distributing:
            raise ValueError("finite distributing material policy requires finite attachment records")
        self._distributing_checkpoint_kind = ("finite_distributing_coupled" if self._distributing_shell_material else
            "finite_distributing_material" if self._distributing_material else "finite_distributing")
        if self._distributing_cyclic:
            self._distributing_checkpoint_kind = ("finite_distributing_cyclic_coupled" if
                self._distributing_shell_material else "finite_distributing_cyclic_material")
        if self._distributing_native:
            self._distributing_checkpoint_kind = ("finite_distributing_cyclic_native_coupled" if
                self._distributing_shell_material else "finite_distributing_cyclic_native_material")
        if groups and finite_groups:
            raise ValueError("reference-only and finite distributing records cannot be mixed")
        reference_groups = groups
        groups = groups or tuple(g.footprint for g in finite_groups)
        if len({g.footprint_id for g in groups}) != len(groups):
            raise ValueError("duplicate distributing footprint ID")
        from ._stiffened_shell_ge_reference_modal import (
            GE_FREE_FREE_MODAL_ONLY_V1, GE_MIXED_FREE_FREE_MODAL_ONLY_V1)
        if (type(definition.modal_boundary_policy) is not str or
                definition.modal_boundary_policy not in ("supported", GE_FREE_FREE_MODAL_ONLY_V1,
                                                         GE_MIXED_FREE_FREE_MODAL_ONLY_V1)):
            raise ValueError("explicit admitted GE modal boundary policy required")
        self._modal_only = definition.modal_boundary_policy in (
            GE_FREE_FREE_MODAL_ONLY_V1, GE_MIXED_FREE_FREE_MODAL_ONLY_V1)
        if self._modal_only and (definition.fixed or definition.lines or definition.footprints):
            raise ValueError("free-free modal-only policy requires empty supports and six-pose points only")
        if (type(definition.bl5_material_policy) is not str or
                definition.bl5_material_policy not in ("elastic", BL5_GENERALIZED_MATERIAL_V1, BL5_COUPLED_J2_MATERIAL_V1, BL5_ELASTIC_NATIVE_FIBRE_V1, BL5_ELASTIC_NATIVE_FIBRE_MIXED_SHELL_V1, BL5_NATIVE_FIBRE_MATERIAL_V1, BL5_NATIVE_FIBRE_MIXED_SHELL_MATERIAL_V1)):
            raise ValueError("explicit admitted BL5 material policy required")
        self._bl5_shell_material = definition.bl5_material_policy == BL5_COUPLED_J2_MATERIAL_V1
        self._bl5_native_mixed_material = definition.bl5_material_policy == BL5_NATIVE_FIBRE_MIXED_SHELL_MATERIAL_V1
        self._bl5_native_mixed_shell = definition.bl5_material_policy in (BL5_ELASTIC_NATIVE_FIBRE_MIXED_SHELL_V1, BL5_NATIVE_FIBRE_MIXED_SHELL_MATERIAL_V1)
        self._bl5_native_material = definition.bl5_material_policy in (BL5_NATIVE_FIBRE_MATERIAL_V1, BL5_NATIVE_FIBRE_MIXED_SHELL_MATERIAL_V1)
        self._bl5_native_fibre = definition.bl5_material_policy in (BL5_ELASTIC_NATIVE_FIBRE_V1, BL5_ELASTIC_NATIVE_FIBRE_MIXED_SHELL_V1, BL5_NATIVE_FIBRE_MATERIAL_V1, BL5_NATIVE_FIBRE_MIXED_SHELL_MATERIAL_V1)
        self._bl5_material = definition.bl5_material_policy in (BL5_GENERALIZED_MATERIAL_V1, BL5_COUPLED_J2_MATERIAL_V1)
        if self._bl5_material and (not definition.lines or definition.footprints or
                definition.q4_pullback_policy != Q4_ALGORITHMIC_PULLBACK_V1):
            raise ValueError("BL5 material policy requires objective Q4 line model")
        panel = definition.shell
        if type(panel) is not ShellPanel:
            raise ValueError("owned Q4 panel required")
        if self._distributing_material and (panel.j2_curve is not None) != self._distributing_shell_material:
            raise ValueError("finite distributing coupled material policy requires matching shell J2")
        if groups and (not panel.q4_elements or panel.s3_elements or
                (panel.j2_curve is not None and not self._distributing_shell_material) or
                definition.q4_pullback_policy != Q4_ALGORITHMIC_PULLBACK_V1 or
                definition.shell_reference_policy != "flat" or definition.bl5_material_policy != "elastic" or
                definition.modal_boundary_policy != "supported" or definition.lines or definition.footprints or
                definition.footprint_chart_policy != CHART_ADMISSION_POLICY or
                definition.retained_scale_policy != "profile"):
            raise ValueError("distributing reference footprints require flat elastic full-map Q4, supported generalized beams, and no lines/rigid footprints/finite policies")
        if groups and (type(definition.points) is not tuple or any(type(p) is not PointConnection for p in definition.points)):
            raise ValueError("typed independent reference point connections required")
        if definition.shell_reference_policy not in ("flat",CURVED_Q4_POINT_REFERENCE_V1,
                                                      CURVED_Q4_POINT_MATERIAL_REFERENCE_V1,
                                                      CURVED_Q4_S3_POINT_REFERENCE_V1):
            raise ValueError("explicit admitted shell reference policy required")
        self._curved_material=definition.shell_reference_policy==CURVED_Q4_POINT_MATERIAL_REFERENCE_V1
        self._curved_s3=definition.shell_reference_policy==CURVED_Q4_S3_POINT_REFERENCE_V1
        self._curved_reference=definition.shell_reference_policy!="flat"
        if (type(definition.retained_scale_policy) is not str or
                definition.retained_scale_policy not in ("profile", "native_force_moment_roles_v1") or
                (definition.retained_scale_policy != "profile" and not self._curved_material)):
            raise ValueError("native retained scaling requires the explicit curved material point policy")
        self._native_retained_scale = definition.retained_scale_policy != "profile"
        if self._curved_reference and ((panel.s3_elements and not self._curved_s3)
                or (self._curved_s3 and (not panel.s3_elements or panel.j2_curve is not None))
                or (panel.j2_curve is not None and not self._curved_material)
                or definition.lines or definition.footprints or self._bl5_material or self._modal_only
                or definition.q4_pullback_policy!=Q4_ALGORITHMIC_PULLBACK_V1):
            raise ValueError("curved material reference requires Q4/full-pose points with algorithmic pullback"
                if self._curved_material else
                "curved reference requires the selected elastic Q4 or Q4/S3 full-pose point policy")
        if (type(definition.q4_pullback_policy) is not str or
                definition.q4_pullback_policy not in (Q4_HISTORICAL_POLICY, Q4_ALGORITHMIC_PULLBACK_V1)):
            raise ValueError("explicit admitted Q4 pullback policy required")
        if definition.footprints and definition.lines:
            raise ValueError("experimental nodal rigid footprints require point-owner-only attachment scope")
        if definition.q4_pullback_policy == Q4_ALGORITHMIC_PULLBACK_V1 and (
                (not panel.q4_elements and not (self._bl5_native_mixed_shell or self._curved_s3)) or (definition.lines and (
                    (panel.s3_elements and not self._bl5_native_mixed_shell) or (panel.j2_curve is not None and not (self._bl5_shell_material or self._bl5_native_material)) or
                    any(type(line) is not ExperimentalBL5Line for line in definition.lines)))):
            raise ValueError("algorithmic Q4 pullback requires a Q4-containing point or explicitly admitted Q4 BL5 model")
        shell_model, shell_ids, shell_xyz, shell_index, shell_parts = _shell_reference(
            panel, curved=self._curved_reference, oriented=self._curved_s3)
        # The compiled owner must not consult caller-owned mutable connectivity
        # when selecting material recovery or decoding a later checkpoint.
        panel = ShellPanel(
            tuple(shell_ids), tuple(map(tuple, shell_xyz.tolist())),
            tuple(tuple(cell) for cell in panel.q4_elements),
            panel.thickness_m, panel.young_pa, panel.poisson,
            panel.density_kg_m3,
            tuple(tuple(cell) for cell in panel.s3_elements), panel.j2_curve)
        definition = LinearDefinition(
            definition.profile, panel, tuple(definition.beams),
            tuple(definition.points), tuple(definition.fixed),
            definition.lines, definition.q4_pullback_policy, definition.footprints,
            definition.modal_boundary_policy, definition.footprint_chart_policy,
            definition.bl5_material_policy, definition.shell_reference_policy,
            definition.retained_scale_policy, reference_groups, finite_groups,
            definition.finite_distributing_material_policy, definition.cyclic_reference_actions,
            definition.shell_support_policy)
        beam_ids, beam_index, beam_nodes, beam_parts, beam_count, beam_inertias = _beam_reference(
            definition.beams)
        self._reference_modal_inertias = beam_inertias
        self._reference_modal_mass = None
        self._mixed_reference_modal_mass = None
        self._has_physical_fibre = any(type(element.section) is PhysicalFibreSection
                                      for element, _, _ in beam_parts)
        if self._coordinate_support and self._has_physical_fibre:
            raise ValueError("coordinate shell supports exclude physical-fibre material sections")
        if groups and self._has_physical_fibre and not self._distributing_native:
            raise ValueError("distributing reference footprints exclude physical-fibre sections")
        if self._distributing_native and not self._has_physical_fibre:
            raise ValueError("native distributing material policy requires a physical-fibre section")
        if self._curved_reference and self._has_physical_fibre:
            raise ValueError("curved reference admits generalized elastic beam response only")
        if definition.footprints:
            if (panel.q4_elements and
                    definition.q4_pullback_policy != Q4_ALGORITHMIC_PULLBACK_V1):
                raise ValueError("nodal rigid footprints require explicitly selected full-map Q4")
            if panel.j2_curve is not None or self._has_physical_fibre:
                raise ValueError("experimental nodal rigid footprints admit elastic shell and generalized beam response only")
            if (any(type(v) is not PointConnection for v in definition.points) or
                    type(definition.lines) is not tuple or
                    any(type(v) not in (ExperimentalL1Line, ExperimentalBL5Line)
                        for v in definition.lines)):
                raise ValueError("typed explicit point/line attachment declarations required")
        definition = replace(definition, footprints=validate_footprints(
            definition.footprints, shell_index, shell_xyz,
            panel.q4_elements+panel.s3_elements, beam_index,
            definition.points, definition.lines))
        chart_admission_policy_descriptor(definition.footprint_chart_policy)
        if (definition.footprint_chart_policy == INCREMENT_ORIGIN_CHART_POLICY and
                (not definition.footprints or definition.lines)):
            raise ValueError("increment-origin chart policy requires an elastic nodal footprint model without lines")
        self._footprint_increment_chart = bool(
            definition.footprints and definition.footprint_chart_policy == INCREMENT_ORIGIN_CHART_POLICY)
        self._finite_elastic_footprint = bool(definition.footprints)
        self._footprint_stable_s3 = bool(definition.footprints and panel.s3_elements
                                        and not definition.lines)
        self._beam_families = tuple(dict.fromkeys(
            "PHYSICAL_AXIAL_BIAXIAL_FIBRE" if type(element.section) is PhysicalFibreSection
            else "RESULTANT_ELLIPSOID" for element, _, _ in beam_parts))
        if self._bl5_native_fibre and (not self._has_physical_fibre or not definition.lines or
                (self._bl5_native_mixed_material and panel.j2_curve is None) or
                (not panel.s3_elements if self._bl5_native_mixed_shell else
                 not panel.q4_elements or bool(panel.s3_elements)) or (panel.j2_curve is not None and not self._bl5_native_material) or
                definition.shell_reference_policy != "flat" or definition.footprints or groups or
                (bool(panel.q4_elements) and definition.q4_pullback_policy != Q4_ALGORITHMIC_PULLBACK_V1) or
                any(type(line) is not ExperimentalBL5Line or line.basis not in
                    ("p0", "continuous_p1_coarse") for line in definition.lines)):
            raise ValueError("native-fibre BL5 requires its explicit flat shell material domain, objective Q4 when present, supported director-bond lines and physical fibres")
        if self._has_physical_fibre and definition.lines and not self._bl5_native_fibre:
            raise ValueError("physical-fibre mixed owner currently admits point attachments only")
        shell_count = 6*len(shell_ids)
        reference_length, shell_scale, beam_scale = _reference_scales(
            shell_xyz, beam_nodes, beam_parts, shell_count, beam_count,
            footprint_retained_roles=(self._finite_elastic_footprint or self._bl5_shell_material
                                      or self._bl5_native_material or self._native_retained_scale or self._finite_distributing))
        shell_r, shell_k, shell_local_error = _assemble_shell(
            shell_model, shell_parts, shell_count, shell_scale,
            definition.q4_pullback_policy)
        beam_r, beam_k, beam_local_error = _assemble_beam(
            beam_parts, beam_nodes, beam_index, beam_count, beam_scale)
        reference_error = max(
            _scaled_reference_residual(shell_r, shell_k, shell_scale),
            _scaled_reference_residual(beam_r, beam_k, beam_scale),
            shell_local_error, beam_local_error)
        if reference_error > 32.*np.finfo(float).eps:
            raise ValueError("reference state must be stress free")
        self._reference_length = reference_length
        self._reference_error = reference_error
        self._K = sparse.block_diag((shell_k, beam_k), format="csr")
        self._coordinate_scale = _owned(np.r_[shell_scale, beam_scale])
        self._shell_count = shell_count
        self._beam_count = beam_count
        self._shell_ids = shell_ids
        self._beam_ids = beam_ids
        self._shell_index = shell_index
        self._beam_index = beam_index
        self._beam_nodes = beam_nodes
        self._shell_xyz = shell_xyz
        self._shell_model = shell_model
        self._shell_parts = tuple(shell_parts)
        self._beam_parts = tuple(beam_parts)
        self._definition = definition
        self._cyclic_reference_pattern = None
        if self._distributing_cyclic:
            from ._stiffened_shell_ge_finite_experimental import _nodal_translation_pattern
            pattern = _nodal_translation_pattern(self, definition.cyclic_reference_actions)
            self._cyclic_reference_pattern = pattern
            self._definition = replace(definition, cyclic_reference_actions=tuple(
                NodalAction(action.domain, action.node, tuple(float(v) for v in action.wrench))
                for action in definition.cyclic_reference_actions))
        distributing_compiled = []
        used_shell, used_hosts, used_beam = set(), set(), set()
        for group in groups:
            compiled = _compile_reference_geometry(group, shell_parts=self._shell_parts,
                shell_index=shell_index, shell_xyz=shell_xyz, beam_index=beam_index,
                beam_nodes=beam_nodes, shell_count=shell_count, global_dimension=self._K.shape[0])
            if (used_shell.intersection(compiled.shell_node_ids) or used_hosts.intersection(group.host_element_ids)
                    or group.beam_node_id in used_beam):
                raise ValueError(f"overlapping distributing footprint node/host/beam mapping: footprint={group.footprint_id}")
            for point in definition.points:
                if point.shell_node in compiled.shell_node_ids or point.beam_node == group.beam_node_id:
                    raise ValueError(f"distributing footprint conflicts with point shell={point.shell_node} beam={point.beam_node}: footprint={group.footprint_id}")
            for previous in distributing_compiled:
                a, b = previous.definition.bounds_xy_m, group.bounds_xy_m
                if min(a[1], b[1]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[2], b[2]):
                    raise ValueError(f"overlapping distributing footprint areas: footprint={group.footprint_id}")
            used_shell.update(compiled.shell_node_ids)
            used_hosts.update(group.host_element_ids)
            used_beam.add(group.beam_node_id)
            distributing_compiled.append(compiled)
        if (type(definition.lines) is not tuple or
                any(type(line) not in (ExperimentalL1Line, ExperimentalBL5Line)
                    for line in definition.lines)):
            raise ValueError("typed experimental weak lines required")
        self._finite_bl5 = bool(definition.lines) and all(
            type(line) is ExperimentalBL5Line for line in definition.lines
        ) and (panel.j2_curve is None or self._bl5_shell_material or self._bl5_native_material) and (not self._has_physical_fibre or self._bl5_native_fibre)
        if definition.lines and definition.q4_pullback_policy == Q4_ALGORITHMIC_PULLBACK_V1 and not self._finite_bl5:
            raise ValueError("algorithmic Q4 line pullback requires admitted generalized BL5 beams")
        if self._bl5_material and not self._finite_bl5:
            raise ValueError("BL5 material policy requires admitted Q4 shells and generalized beams")
        if self._bl5_shell_material and panel.j2_curve is None:
            raise ValueError("coupled BL5 material policy requires shell J2")
        identity_payload = {
            "profile": PROFILE, "shell_ids": shell_ids,
            "shell_xyz": shell_xyz.tolist(), "shell_cells": panel.q4_elements,
            "shell_s3_cells": panel.s3_elements,
            "thickness": panel.thickness_m, "E": panel.young_pa,
            "nu": panel.poisson, "density": panel.density_kg_m3,
            "beams": [item.sha256 for item in definition.beams],
            "points": [(v.shell_node, v.beam_node) for v in definition.points],
            "fixed": [(v.domain, v.node, v.component) for v in definition.fixed],
        }
        if self._distributing_reference:
            identity_payload["distributing_reference"] = dict(law=DISTRIBUTING_REFERENCE_LAW,
                records=[(g.footprint_id, g.host_element_ids, g.beam_node_id, g.bounds_xy_m) for g in groups],
                compiled=[g.identity for g in distributing_compiled],
                policies=(definition.q4_pullback_policy, definition.shell_reference_policy,
                          definition.bl5_material_policy, definition.modal_boundary_policy,
                          definition.footprint_chart_policy), rows="common identity reference-spatial u/omega")
        finite_poses = tuple(FiniteDistributingPose(g) for g in distributing_compiled) if self._finite_distributing else ()
        if self._finite_distributing:
            identity_payload["distributing_finite"] = dict(law=DISTRIBUTING_FINITE_LAW,
                geometry=[g.identity for g in distributing_compiled], poses=[p.identity for p in finite_poses],
                retained_scales="native_force_moment_roles_v1", checkpoint_kind="finite_distributing",
                policy="flat_elastic_q4_generalized_monotone_translation_force_v1")
            if self._distributing_material:
                identity_payload["distributing_finite_material"] = dict(
                    policy=definition.finite_distributing_material_policy,
                    shell_rotation_numerics="GE_BEAM3_G3C_ISOLATED_SO3_NUMERICS_V1",
                    checkpoint_kind=self._distributing_checkpoint_kind,
                    histories="accepted_and_fixed_predecessor_v1")
                if self._distributing_cyclic:
                    identity_payload["distributing_finite_material"]["force_history"] = dict(
                        admission="finite_signed_changed_factors_fixed_v1",
                        zero_load_norm="immutable_unit_factor_reference_pattern_v1",
                        reference_pattern=self._cyclic_reference_pattern.tolist())
        if self._bl5_shell_material:
            identity_payload["bl5_retained_scale_policy"] = "native_force_moment_roles_v1"
        if self._curved_reference:
            identity_payload["shell_reference_policy"] = definition.shell_reference_policy
        if self._native_retained_scale:
            identity_payload["retained_scale_policy"] = definition.retained_scale_policy
        if self._bl5_material or self._bl5_native_fibre:
            identity_payload["bl5_material_policy"] = definition.bl5_material_policy
        if self._bl5_native_material:
            identity_payload["bl5_native_material_retained_scales"] = "native_force_moment_roles_v1"
        if panel.j2_curve is not None:
            identity_payload["shell_j2_curve"] = panel.j2_curve.as_dict()
            if panel.s3_elements:
                # Experimental direct-coordinate histories must not silently
                # resume with the new objective local constitutive map.
                identity_payload["shell_s3_material_policy"] = S3_J2_PULLBACK_POLICY
        if definition.q4_pullback_policy != Q4_HISTORICAL_POLICY:
            identity_payload["shell_q4_pullback_policy"] = definition.q4_pullback_policy
        if definition.lines:
            identity_payload["lines"] = [line.descriptor() for line in
                                         definition.lines]
        if definition.footprints:
            identity_payload["nodal_rigid_footprints"] = [
                group.descriptor() for group in definition.footprints]
            identity_payload["footprint_retained_scale_policy"] = retained_scale_policy_descriptor()
            identity_payload["footprint_chart_admission_policy"] = chart_admission_policy_descriptor(definition.footprint_chart_policy)
        if self._footprint_stable_s3:
            identity_payload["footprint_finite_numerical_policy"] = stable_s3_policy_descriptor()
        if self._modal_only:
            identity_payload["modal_boundary_policy"] = definition.modal_boundary_policy
        if self._coordinate_support:
            identity_payload['shell_support_policy'] = dict(policy=NATIVE_SHELL_COORDINATE_SUPPORT_V1,
                equations='S Log(Exp(theta) A)=0',branch='both_local_and_original_principal_norm_less_than_pi_minus_1e-7',
                rows='parameter_covectors_with_complete_residual_curvature',
                s3_arithmetic='existing_displacement_centred_quaternion_RminusI' if panel.s3_elements else 'unused')
        self._identity = sha256(json.dumps(
            identity_payload, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()
        self._connections = []
        self._joint_slots = []
        incidence = list(definition.points)
        bindings = []
        footprint_sources = {}
        for group in definition.footprints:
            indices = tuple(range(len(incidence), len(incidence)+len(group.shell_nodes)))
            bindings.append((group, indices))
            for index, node in zip(indices, group.shell_nodes):
                incidence.append(PointConnection(node, group.beam_node))
                footprint_sources[index] = f"footprint={group.footprint_id} "
        self._point_incidence = tuple(incidence)
        self._footprint_bindings = tuple(bindings)
        constraint_rows = []
        row_sources = []
        if not self._point_incidence and not definition.lines and not groups:
            raise ValueError("explicit native connection required")
        seen_points = set()
        for point_index, tie in enumerate(self._point_incidence):
            if (type(tie) is not PointConnection or
                    tie.shell_node not in shell_index or tie.beam_node not in beam_index):
                raise ValueError("known point connection nodes required")
            pair = (tie.shell_node, tie.beam_node)
            if pair in seen_points:
                raise ValueError("duplicate experimental point connection")
            seen_points.add(pair)
            shell_slot = 6*shell_index[tie.shell_node]
            beam_slot = shell_count+6*beam_index[tie.beam_node]
            positions = np.array((shell_xyz[shell_index[tie.shell_node]],
                                  beam_nodes[tie.beam_node][0]))
            frames = np.array((np.eye(3), beam_nodes[tie.beam_node][1]))
            joint = RigidPoseJoint(positions, frames)
            evaluation = joint.evaluate(positions, frames, np.zeros(6))
            if np.max(np.abs(evaluation.constraints)) > 1e-10:
                raise ValueError("point connection is not stress free")
            slots = np.r_[np.arange(shell_slot, shell_slot+6),
                          np.arange(beam_slot, beam_slot+6)]
            constraint_rows.append((slots, evaluation.constraint_jacobian))
            row_sources.extend(
                footprint_sources.get(point_index, "")+
                f"point shell={tie.shell_node} beam={tie.beam_node} row={row}"
                for row in range(6))
            self._connections.append(joint)
            self._joint_slots.append(slots)
        c_row, c_col, c_data = [], [], []
        for i, (slots, jacobian) in enumerate(constraint_rows):
            c_row.extend(np.repeat(np.arange(6*i, 6*i+6), 12))
            c_col.extend(np.tile(slots, 6))
            c_data.extend(jacobian.ravel())
        count = self._K.shape[0]
        self._C = sparse.coo_matrix((c_data, (c_row, c_col)),
                                    shape=(6*len(constraint_rows), count)).tocsr()
        self._distributing_bindings = tuple((compiled, 6*(len(constraint_rows)+i))
                                            for i, compiled in enumerate(distributing_compiled)) if self._distributing_reference else ()
        if groups:
            self._C = sparse.vstack((self._C, *(_scatter_reference(g, count) for g in distributing_compiled)), format="csr")
            self._C.sum_duplicates()
            self._C.eliminate_zeros()
            self._C.sort_indices()
            if not np.isfinite(self._C.data).all():
                raise ValueError("finite combined distributing reference operator required")
            row_sources.extend(f"footprint={g.definition.footprint_id} distributing reference row={row}"
                               for g in distributing_compiled for row in range(6))
        if definition.lines:
            line_c, self._line_samples, self._line_terms = _compile_weak_lines(
                definition.lines, shell_ids=shell_ids, shell_xyz=shell_xyz,
                shell_cells=panel.q4_elements+panel.s3_elements,
                thickness=panel.thickness_m, beam_ids=beam_ids,
                beam_nodes=beam_nodes, beam_parts=beam_parts,
                shell_count=shell_count, beam_count=beam_count, with_terms=True)
            self._C = sparse.vstack((self._C, line_c), format="csr")
            row_sources.extend(
                f"line {index} shell={line.shell_nodes[0]}..{line.shell_nodes[-1]} row={row}"
                for index, line in enumerate(definition.lines)
                for row in range(_weak_line_row_count(line)))
        else:
            self._line_samples = ()
            self._line_terms = ()
        self._constraint_scale = _owned(
            np.asarray(np.abs(self._C)@self._coordinate_scale).reshape(-1))
        if np.any(self._constraint_scale <= 0.):
            raise ValueError("nonzero physical constraint row scales required")
        fixed = set()
        for support in definition.fixed:
            if (type(support) is not FixedCoordinate or
                    support.domain not in ("shell", "beam") or
                    type(support.component) is not int or
                    not 0 <= support.component < 6):
                raise ValueError("known physical fixed coordinate required")
            index = shell_index if support.domain == "shell" else beam_index
            if support.node not in index:
                raise ValueError("known physical fixed node required")
            start = 0 if support.domain == "shell" else shell_count
            fixed.add(start+6*index[support.node]+support.component)
        self._fixed = np.array(sorted(fixed), dtype=int)
        self._free = np.setdiff1d(np.arange(count), self._fixed)
        if self._distributing_cyclic:
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                reference_work = self._cyclic_reference_pattern[self._free]*self._coordinate_scale[self._free]
            if not np.isfinite(reference_work).all() or not np.any(reference_work):
                raise ValueError("cyclic history requires representable nonzero free-coordinate reference actions")
        if (not len(self._fixed) and not self._modal_only) or not len(self._free):
            raise ValueError("physical supports and free coordinates required")
        # This exact five-row layout cannot restrain all six rigid motions
        # of an otherwise free native beam. Other lines or beam supports may
        # supply the missing restraint and are not classified here.
        self._unsupported_modes = (
            ("single-support BL5 leaves a free native beam rigid mode",)
            if (len(definition.lines) == 1 and
                type(definition.lines[0]) is ExperimentalBL5Line and
                definition.lines[0].basis == "p0" and
                len(definition.lines[0].beam_halfcells) ==
                definition.lines[0].multiplier_span and
                not self._connections and
                not any(support.domain == "beam" for support in definition.fixed))
            else ())
        _require_independent_supported_rows(
            self._C, self._free, self._coordinate_scale,
            self._constraint_scale, tuple(row_sources))
        self._A = sparse.bmat([
            [self._K[self._free, :][:, self._free], self._C[:, self._free].T],
            [self._C[:, self._free], None]], format="csc")
        if self._A.shape[0] < 1 or self._A.nnz == 0:
            raise ValueError("nonempty constrained mixed operator required")
        self._operation_lock = RLock()
        self._writing = False
        self._commit_lock = RLock()
        self._accepted = None
        self._finite_driver = None
        self._generation = 0
        self._closed = False
        self._linear_factor = None
        self._linear_factorizations = 0
        self._linear_refinements = 0
        self._finite_distributing_bindings = tuple((
            PreparedFiniteDistributingAttachment._from_compiled(self, compiled, pose),
            6*(len(constraint_rows)+i)) for i,(compiled,pose) in enumerate(zip(distributing_compiled,finite_poses)))
        if self._modal_only:
            from ._stiffened_shell_ge_reference_modal import _admit
            _admit(self, mixed=definition.modal_boundary_policy == GE_MIXED_FREE_FREE_MODAL_ONLY_V1)

    def _require_open(self):
        if self._closed:
            raise RuntimeError("experimental GE analysis is closed")

    def _require_stateful_scope(self):
        if self._modal_only:
            raise ValueError("free-free modal-only policy does not admit static, finite, restore or chart state writes")

    def _require_distributing_reference_only(self, operation):
        if self._distributing_reference:
            raise ValueError(f"distributing footprints admit reference-linear nodal dead loads/checkpoints only; {operation} is unsupported")

    def _require_finite_distributing_scope(self, operation):
        if self._finite_distributing:
            raise ValueError(f"finite distributing footprints do not yet admit {operation}")

    def _require_distributing_linear_checkpoint(self, raw):
        if not self._distributing_reference or raw.startswith(BINARY_CHECKPOINT_MAGIC):
            return
        if len(raw) > MAX_JSON_CHECKPOINT_BYTES:
            raise ValueError("bounded distributing reference-linear checkpoint required")
        try:
            value = json.loads(raw.decode("ascii"))
        except (ValueError, UnicodeDecodeError, RecursionError) as error:
            raise ValueError("distributing footprints require a reference-linear checkpoint; wrappers/finite kinds are unsupported") from error
        if type(value) is not dict or value.get("schema") != CHECKPOINT:
            raise ValueError("distributing footprints require a reference-linear checkpoint; wrappers/finite kinds are unsupported")

    @property
    def capabilities(self):
        if self._finite_distributing:
            if self._distributing_material:
                return dict(profile=PROFILE, status="EXPERIMENTAL_NOT_QUALIFIED",
                    shell="isotropic J2 flat Cartesian full-map Q4" if self._distributing_shell_material else
                          "elastic flat Cartesian full-map Q4",
                    beam=("physical axial/biaxial fibre and generalized material B3 with native retained coordinates"
                        if self._distributing_native else "generalized material B3 with native retained coordinates"),
                    beam_section_families=self._beam_families,
                    attachment=("objective finite area-mean/director/projected-polar distributing pose",),
                    distributing_footprint_law=DISTRIBUTING_FINITE_LAW,
                    material_policy=self._definition.finite_distributing_material_policy,
                    shell_rotation_numerics="GE_BEAM3_G3C_ISOLATED_SO3_NUMERICS_V1",
                    analysis=("nonlinear_force", "experimental_fixed_displacement"), loads=("nodal dead translation forces",),
                    displacement_control="fixed signed physical translational targets from accepted fixed-force seed",
                    displacement_checkpoint="finite_distributing_displacement",
                    fixed_step_max_backtracks=16,
                    increment_control=(("fixed",) if self._distributing_cyclic else
                        ("fixed", "experimental_adaptive_halving")),
                    adaptive_max_backtracks=None if self._distributing_cyclic else 12,
                    adaptive_targets=("not admitted for cyclic history" if self._distributing_cyclic else
                        "strictly increasing positive force factors"),
                    force_history=("finite signed changed fixed-step factors" if self._distributing_cyclic else
                        "strictly increasing positive force factors"),
                    zero_load_normalization=("immutable unit-factor reference pattern" if self._distributing_cyclic else
                        "actual applied force"),
                    checkpoints=(self._distributing_checkpoint_kind, "finite_distributing_displacement"),
                    recovery="accepted constitutive states and evaluated spatial connection actions",
                    excluded=("reference-linear solve", "chart rebase", "adaptive/event displacement", "arc controllers",
                        "external/staged loads") + (() if self._distributing_native else ("physical fibres",)) +
                        ("S3", "modes", "buckling", "qualified distributing footprint"))
            return dict(profile=PROFILE, status="EXPERIMENTAL_NOT_QUALIFIED",
                shell="elastic flat Cartesian full-map Q4", beam="generalized B3 with native retained coordinates",
                beam_section_families=self._beam_families,
                attachment=("objective finite area-mean/director/projected-polar distributing pose",),
                distributing_footprint_law=DISTRIBUTING_FINITE_LAW,
                analysis=("nonlinear_force", "experimental_fixed_displacement"), loads=("nodal dead translation forces",),
                displacement_control="fixed signed physical translational targets from accepted fixed-force seed",
                displacement_checkpoint="finite_distributing_displacement",
                increment_control=("fixed", "experimental_adaptive_halving"),
                adaptive_max_backtracks=12,
                adaptive_targets="strictly increasing positive force factors",
                checkpoints=("finite_distributing", "finite_distributing_displacement"),
                recovery="cached evaluated exerted spatial actions at current physical origins",
                excluded=("reference-linear solve", "chart rebase", "adaptive/event displacement", "arc controllers",
                    "external/staged loads", "material yielding", "modes", "buckling", "qualified distributing footprint"))
        if self._distributing_reference:
            return dict(profile=PROFILE, status="EXPERIMENTAL_NOT_QUALIFIED",
                shell="elastic flat Cartesian full-map Q4", beam="generalized B3 retained, virgin elastic reference response",
                beam_section_families=self._beam_families,
                attachment=(("reference point pose",) if self._connections else ())+
                           ("experimental area-mean/director/polar-yaw reference-linear distributing footprint",),
                distributing_footprint_law=DISTRIBUTING_REFERENCE_LAW,
                distributing_footprint_recovery="physical exerted reference-spatial actions, shell area-centroid and actual beam-node origins",
                analysis=("reference_linear",), loads=("nodal dead actions",), checkpoints=("reference_linear",),
                q4_force_map=self._definition.q4_pullback_policy,
                excluded=("finite solve", "chart rebase", "arc/displacement/globalization controllers",
                          "external load wrappers", "material updates", "modes", "buckling",
                          "qualified distributing footprint"))
        from ._stiffened_shell_ge_distributed_load import admits_owner as admits_distributed
        from ._stiffened_shell_ge_body_load import admits_owner as admits_body
        if self._modal_only:
            from ._stiffened_shell_ge_reference_modal import GE_MIXED_FREE_FREE_MODAL_ONLY_V1
            modal = (self.mixed_reference_modal_capabilities
                     if self._definition.modal_boundary_policy == GE_MIXED_FREE_FREE_MODAL_ONLY_V1
                     else self.reference_modal_capabilities)
            return dict(modal, profile=PROFILE,
                        modal_boundary_policy=self._definition.modal_boundary_policy,
                        accepted_state_writes=False)
        attachment = (("reference point pose",) if self._connections else ())
        if self._line_samples:
            attachment += tuple(dict.fromkeys(
                "experimental L1 weak-line candidate"
                if type(line) is ExperimentalL1Line else
                "experimental BL5 director-bond candidate"
                for line in self._definition.lines))
        shell_plastic = self._definition.shell.j2_curve is not None
        line_limits = tuple(dict.fromkeys(
            "nonmatching L1 line mapping" if type(line) is ExperimentalL1Line
            else "curved/reversed BL5 line mapping"
            for line in self._definition.lines))
        excluded = ((("shell J2 plastic",) if not shell_plastic else ()) +
                    ("nonpositive finite load factors", "line", "footprint",
                     "modes", "buckling")
                    if not self._line_samples else
                    ((() if self._bl5_shell_material else ("shell J2 plastic",)) + ("physical-fibre B3 finite",
                     "qualified bonded line", "footprint",
                     "modes", "buckling"))+line_limits)
        if self._finite_bl5 and not self._bl5_material:
            excluded += ("generalized B3 plastic finite BL5",)
        if (self._definition.shell.s3_elements and shell_plastic and
                not self._line_samples):
            excluded += ("qualified S3 J2 finite-rotation response",)
        capabilities = {"profile": PROFILE, "status": "EXPERIMENTAL_NOT_QUALIFIED",
                "shell": (("experimental isotropic Q4/S3-V2D J2"
                           if self._definition.shell.s3_elements else
                           "experimental isotropic Q4 J2") if shell_plastic else
                          "elastic flat Q4/S3"),
                "beam": ("native B3 retained, experimental physical-fibre histories"
                         if self._has_physical_fibre else
                         "generalized B3 retained" if self._line_samples else
                         "generalized B3 retained, including experimental yielding"),
                "beam_section_families": self._beam_families,
                "attachment": attachment,
                "point_recovery": "accepted global spatial connection actions; reference-linear or current ports",
                "body_acceleration": (() if not admits_body(self) else
                    (("experimental selected-nodal rigid footprint uniform spatial acceleration; elastic response of generalized sections; flat Q4 and increment-origin charts; explicit physical beam inertia and shell density; GEBodyLoad",)
                     if self._definition.footprints else
                     ("experimental uniform spatial acceleration; explicit physical beam inertia and shell density; reference-linear or conservative finite work with objective Q4 pullback; GEBodyLoad",))),
                "beam_distributed_loads": (() if not admits_distributed(self) else
                    ("experimental element-addressed fixed-spatial force/couple per reference arclength; native quadrature; couples nonconservative; GEBeamDistributedLoad",)),
                "staged_beam_distributed_loads": (() if not admits_distributed(self) else
                    ("experimental GEBeamDistributedProgramme; absolute signed force/couple densities at positive forward positions; point or objective elastic Q4 BL5 owner; zero-load convergence unresolved under current equilibrium assessment; not qualified",)),
                "shell_pressure_loads": (() if self._line_samples or self._definition.footprints else
                    ("experimental GEShellPressureLoad; shell IDs are Q4 then S3, one-based; reference-linear native dead pressure; finite Q4 dead and Q4/S3 interpolation-surface current-area follower pressure; finite native reference_dead S3 is unsupported; explicit reference_dead_director_v2 admits a distinct virgin-elastic S3 director-enhanced dead surface with relative tilt below 0.9*pi, not qualified",)),
                "staged_dead_loads": (() if self._line_samples and not (
                    self._finite_bl5 and self._definition.q4_pullback_policy == Q4_ALGORITHMIC_PULLBACK_V1) else
                    ("experimental immutable piecewise-affine translation forces; forward programme position",)),
                "zero_load_normalization": (() if not self._definition.points or self._definition.lines or
                    self._definition.footprints or self._definition.distributing_footprints or
                    self._definition.finite_distributing_footprints or
                    self._definition.shell_reference_policy != "flat" else
                    ("experimental staged opt-in zero_load_reference_endpoint; exact-zero loads only; frozen physical endpoint/free work; separate checkpoint identity; not qualified",)),
                "experimental_line_spaces": tuple(
                    line.multiplier_space for line in self._definition.lines),
                "q4_force_map": self._definition.q4_pullback_policy,
                "bl5_material_policy": self._definition.bl5_material_policy,
                "increment_control": (() if self._line_samples and not self._finite_bl5 else
                    ("fixed", "experimental_adaptive_halving")),
                "rotation_chart": ("unchanged reference-linear chart"
                                   if self._line_samples and not self._finite_bl5 else
                                   "experimental owned shell anchors; full fixed/free rotational triples only"),
                "analysis": (("reference_linear", "experimental_coupled_j2_material_bl5" if self._bl5_shell_material else "experimental_generalized_material_bl5" if self._bl5_material else "experimental_elastic_finite_bl5")
                             if self._finite_bl5 else
                             ("reference_linear",) if self._line_samples else
                             ("reference_linear", "experimental_native_fibre_finite_point")
                             if self._has_physical_fibre else
                             ("reference_linear", "experimental_elastic_finite_point",
                              "experimental_generalized_plastic_finite_point") +
                             (("experimental_q4_s3_j2_finite_point",)
                              if shell_plastic and self._definition.shell.s3_elements
                              else ("experimental_q4_j2_finite_point",)
                              if shell_plastic
                              else ())),
                "excluded": excluded}
        if self._bl5_native_fibre:
            capabilities.update(analysis=("reference_linear", "experimental_elastic_native_fibre_bl5"),
                native_bl5_material_domain="virgin unchanged history, every native station/fibre elastic-interior with positive yield margin",
                native_bl5_checkpoint="existing native-material binary schema; explicit model/line/material-policy identity",
                excluded=tuple(v for v in excluded if v != "physical-fibre B3 finite"),
                open_domains=("plastic native BL5", "S3 native BL5", "consumer integration", "scale qualification"))
            if self._bl5_native_mixed_shell:
                capabilities.update(analysis=("reference_linear", "experimental_elastic_native_fibre_mixed_shell_bl5"),
                    native_bl5_checkpoint="existing stable native-material binary schema; explicit mixed-shell model/line/material-policy identity",
                    open_domains=("plastic native BL5", "plastic S3 native BL5", "consumer integration", "scale qualification"))
            if self._bl5_native_material:
                capabilities.update(analysis=("reference_linear", "experimental_native_fibre_material_bl5"),
                    retained_coordinate_scale_policy="native_force_moment_roles_v1",
                    finite_trial_proposal="experimental_fixed_origin_fibre_boundary_v1; actual strict merit; unchanged candidate budget",
                    native_bl5_material_domain="existing physical-fibre and generalized constitutive histories; elastic or isotropic Q4 J2 shell",
                    native_bl5_checkpoint="existing native-material/native-shell binary schema; explicit model/line/material-policy identity",
                    excluded=tuple(v for v in capabilities["excluded"] if v not in ("shell J2 plastic", "generalized B3 plastic finite BL5")) +
                        ("native material BL5 staged/external loads", "native material BL5 adaptive displacement control", "native material BL5 spectra", "native material BL5 chart rebase"),
                    open_domains=("plastic S3 native BL5", "native material BL5 chart rebase", "consumer integration", "scale qualification"))
                for key in ("body_acceleration", "beam_distributed_loads", "staged_beam_distributed_loads", "shell_pressure_loads", "staged_dead_loads"):
                    capabilities[key] = ()
        if self._bl5_native_mixed_material:
            capabilities.update(analysis=("reference_linear", "experimental_native_fibre_mixed_shell_j2_bl5"),
                native_bl5_material_domain="existing physical-fibre and generalized histories; isotropic Q4/S3 J2 shell",
                finite_trial_proposal="experimental_fixed_origin_native_and_q4_s3_boundary_v1; actual strict merit; unchanged candidate budget",
                open_domains=("native material BL5 chart rebase", "consumer integration", "scale qualification"))
        if self._definition.footprints:
            capabilities["attachment"] = (
                (("reference point pose",) if self._definition.points else ())+
                ("experimental selected-node rigid footprint (six-channel pose)",)+
                tuple(v for v in attachment if "line" in v))
            capabilities["excluded"] = tuple(v for v in excluded if v != "footprint")+(
                "plastic footprint", "physical-fibre footprint", "distributing footprint",
                "continuous-area footprint", "qualified footprint", "off-node beam footprint station")
            capabilities["analysis"] = (
                ("reference_linear", "experimental_elastic_finite_nodal_footprint")
                if not self._line_samples or self._finite_bl5 else ("reference_linear",))
            capabilities["footprint_retained_scale_policy"] = retained_scale_policy_descriptor()
            capabilities["footprint_chart_admission_policy"] = chart_admission_policy_descriptor(self._definition.footprint_chart_policy)
            capabilities["excluded"] += (("nodal-footprint finite charts with partial rotational supports",)
                if self._footprint_increment_chart else ("nodal-footprint chart rebase and rebased checkpoint restore",))
        if self._footprint_stable_s3:
            capabilities["footprint_finite_numerical_policy"] = stable_s3_policy_descriptor()
        if self._bl5_native_material:
            capabilities["displacement_control"] = "experimental_native_fibre_bl5_fixed_translation_v1"
            if self._bl5_native_mixed_material:
                capabilities["displacement_control"] = "experimental_native_fibre_bl5_mixed_shell_j2_fixed_translation_v1"
            capabilities["displacement_control_seed"] = "ordinary nonadaptive accepted force endpoint; no chart/event/adaptive controller"
            from ._stiffened_shell_affine_endpoint import is_native_material_bl5_arc
            if is_native_material_bl5_arc(self) and getattr(self,"_external_context",None) is None:
                capabilities["arc_length"] = "experimental_native_fibre_bl5_material_tangent_plane_v1"
                capabilities["arc_metric"] = (
                    "full-frame PhysicalChordMetric; shell normal spin contributes 2*w*(1-cos(theta)); "
                    "not a position/director quotient metric; retained variables and multipliers excluded from metric only")
                capabilities["arc_metric_opt_in"] = (
                    "NativeDirectorChordMetric: explicit flat native material BL5 shell position/director "
                    "and beam full-frame metric; SOURCE CANDIDATE UNVERIFIED; fixed shell-then-beam ownership; "
                    "shell normal spin excluded from metric only; open director hemisphere and nonsingular "
                    "unrebased shell chart; distinct metric history/restart identity; no capacity qualification")
                capabilities["arc_length_limits"] = (
                    "source candidate; fixed unrebased ordinary/fixed-translation seed; affine nodal dead translations; "
                    "fixed physical metric and free translation orientation; accepted-prefix restart; "
                    "no adaptive/event/staged/external seed, chart rebase, branch switching or capacity qualification")
        from ._stiffened_shell_affine_endpoint import is_native_point_material
        if is_native_point_material(self):
            capabilities["displacement_control"] = (
                "experimental_native_point_shell_j2_s3_fixed_translation_v1"
                if self._definition.shell.s3_elements and self._definition.shell.j2_curve is not None else
                "experimental_native_point_fibre_elastic_s3_fixed_translation_v1"
                if self._definition.shell.s3_elements and self._has_physical_fibre else
                "experimental_native_point_elastic_s3_fixed_translation_v1"
                if self._definition.shell.s3_elements else
                "experimental_native_point_coupled_j2_fixed_translation_v1"
                if self._definition.shell.j2_curve is not None else
                "experimental_native_point_material_fixed_translation_v1")
            capabilities["displacement_control_seed"] = (
                "ordinary unrebased nodal-dead-translation force seed; fixed or bounded adaptive targets; "
                "native displacement chart continuation; no staged/external seed, event policy or arc qualification")
            capabilities["adaptive_displacement_control"] = "experimental_native_point_translation_halving_v1"
            capabilities["displacement_chart_rebase"] = "experimental_native_point_material_chart_v1"
            if getattr(self,"_external_context",None) is None:
                capabilities["arc_length"] = "experimental_native_point_material_tangent_plane_v1"
                capabilities["arc_length_limits"] = (
                    "affine nodal dead translations; fixed physical metric and free translation orientation; "
                    "ordinary/displacement endpoint or compatible arc prefix; no branch switching, "
                    "arc chart rebase or limit-point qualification")
        if self._bl5_shell_material:
            capabilities["displacement_control"] = "experimental_nodal_translation_seeded_v1"
            if getattr(self,"_external_context",None) is None:
                capabilities["arc_length"] = "experimental_fixed_mesh_affine_nodal_dead_translation_v1"
            capabilities["selected_finite_recovery"] = "experimental_cached_nodes_material_v1"
        if self._curved_reference:
            from ._stiffened_shell_stress_free_reference import POLICY, MATERIAL_POLICY
            capabilities["stress_free_reference_construction"] = MATERIAL_POLICY if self._curved_material else POLICY
            capabilities["shell"] = ("experimental isotropic J2 planar Q4 facets on curved reference"
                if self._definition.shell.j2_curve is not None else
                "experimental elastic planar Q4/S3 facets on curved reference"
                if self._curved_s3 else
                "experimental elastic planar Q4 facets on curved reference")
            capabilities["beam"] = ("generalized B3 retained, experimental yielding/hardening"
                if self._curved_material else "generalized B3 retained, elastic response only")
            for key in ("body_acceleration","beam_distributed_loads","staged_beam_distributed_loads",
                        "shell_pressure_loads","staged_dead_loads"):
                capabilities[key] = ()
            capabilities["curved_reference_policy"] = self._definition.shell_reference_policy
            if self._curved_material:
                capabilities["displacement_control"] = "experimental_curved_material_point_translation_v1"
                capabilities["staged_dead_loads"] = (
                    "experimental curved-material point piecewise-affine nodal translation forces; forward programme position",)
                capabilities["fixed_step_max_backtracks"] = 16
                capabilities["retained_coordinate_scale_policy"] = (
                    "native_force_moment_roles_v1" if self._native_retained_scale else
                    "legacy_repeated_wrench_v1")
                capabilities["adaptive_max_backtracks"] = 12
                capabilities["generalized_station_recovery"] = "cached_fixed_predecessor_v2; older snapshots omit last-increment records"
            if getattr(self,"_external_context",None) is None:
                capabilities["arc_length"] = ("experimental_curved_material_point_tangent_plane_v1"
                    if self._curved_material else "experimental_curved_elastic_point_tangent_plane_v1")
            capabilities["excluded"] += ("curved reference spectra",
                "curved reference external loads")
            if not self._curved_material:
                capabilities["excluded"] += ("curved reference material yielding",)
        from ._stiffened_shell_ge_pressure_load import VECTOR_EQUILIBRIUM_POLICY
        capabilities["shell_pressure_equilibrium_policies"] = (
            ("component_relative_v1", VECTOR_EQUILIBRIUM_POLICY)
            if capabilities["shell_pressure_loads"] else ())
        capabilities["selective_rotation_chart_rebase"] = (
            "opt-in preserve_supported_charts=True; fixed force-controlled native Q4/S3 point state; "
            "reset rotationally free nodes only; supported anchor/coordinates retained exactly; "
            "Q4 requires existing explicit algorithmic pullback policy; pure S3 uses existing variational rows; excludes displacement/arc/line/footprint/adaptive/staged/external and modal-only routes; "
            "no support-equation transport or full F06 qualification") if (
                not self._definition.lines and not self._definition.footprints and not self._modal_only and
                not self._bl5_native_material and not self._distributing_reference and not self._finite_distributing and
                (not self._definition.shell.q4_elements or self._definition.q4_pullback_policy == Q4_ALGORITHMIC_PULLBACK_V1)) else "not admitted"
        if self._coordinate_support:
            capabilities['shell_support_policy']=NATIVE_SHELL_COORDINATE_SUPPORT_V1
            capabilities['shell_support_equations']='original principal Log components; parameter covectors, physical dual reactions; experimental elastic point force route only'
            capabilities['beam']='generalized B3 retained; elastic response only for coordinate shell supports'
            capabilities['analysis']=('experimental_elastic_finite_point',)
            capabilities['increment_control']=('fixed',)
            capabilities['rotation_chart']='experimental original principal Log supports through full/selective shell rebase; local and original principal branches bounded'
            capabilities['zero_load_normalization']=()
            for key in ('body_acceleration','beam_distributed_loads','staged_beam_distributed_loads','shell_pressure_loads','staged_dead_loads'):
                capabilities[key]=()
            capabilities['displacement_control']='not admitted for coordinate shell supports'
            capabilities['arc_length']='not admitted for coordinate shell supports'
            capabilities['coordinate_support_exclusions']=('linear','spectral','adaptive','displacement','arc','external','staged','plastic')
        from ._stiffened_shell_f06_increment_arithmetic import POLICY as COMPENSATED_POLICY
        if any(getattr(e.operator,'arithmetic_policy',None)==COMPENSATED_POLICY for e,_,_ in self._beam_parts):
            from ._stiffened_shell_ge_finite_experimental import supports_paired_fibre_adaptive, supports_mixed_material_adaptive
            capabilities['increment_control'] = (('fixed', 'experimental_adaptive_halving')
                if supports_paired_fibre_adaptive(self) or supports_mixed_material_adaptive(self) else ('fixed',))
            if supports_paired_fibre_adaptive(self):
                capabilities['adaptive_load_scope'] = 'declared staged dead-load programme only; homogeneous physical-fibre compensated elastic flat Q4 POINT owner; explicit accepted substeps'
                capabilities['adaptive_targets'] = 'strictly increasing positions in the same declared programme; original physical checks and attempt journal'
            elif supports_mixed_material_adaptive(self):
                capabilities['displacement_control']='experimental paired mixed Q4-J2 POINT fixed shell translation; explicit original force_seed_sha256'
                capabilities['adaptive_displacement_control']='experimental implemented signed-halving candidate; unqualified; explicit original force_seed_sha256; fixed shell translation corrector; no event policy'
                capabilities['merit_globalization']='experimental opt-in paired fixed-work quadratic candidate; unqualified; actual original merit and physical acceptance unchanged'
                capabilities['arc_length']='not admitted for paired native material'
                capabilities['adaptive_load_scope'] = 'nodal translation dead-force pattern only; compensated mixed generalized/physical-fibre flat Q4 algorithmic J2 POINT owner; explicit accepted substeps'
                capabilities['adaptive_targets'] = 'positive factors of the unchanged nodal pattern; original physical checks and strict attempt journal; staged conversion excluded'
        return capabilities

    def preflight(self):
        self._require_open()
        details = {"identity": self._identity, "physical_dofs": self._shell_count+
                6*len(self._beam_ids), "retained_dofs": self._beam_count-
                6*len(self._beam_ids), "constraints": self._C.shape[0],
                "matrix_nnz": self._A.nnz,
                "experimental_line_samples": len(self._line_samples),
                "reference_linear_factorizations": self._linear_factorizations,
                "reference_linear_refinements": self._linear_refinements,
                "reference_length_m": self._reference_length,
                "stress_free_scaled_max": self._reference_error,
                "equilibrium_norm": "frozen_work_weighted_l2_plus_active_load_rows",
                "constraint_norm": "reference_row_scaled_max",
                "unsupported_modes": self._unsupported_modes,
                "production_qualified": False}
        if self._finite_distributing:
            details["finite_distributing_footprints"] = tuple(adapter.descriptor()
                for adapter,_rows in self._finite_distributing_bindings)
            details["finite_distributing_local_curvature_nnz_upper_bound"] = sum(
                adapter.descriptor()["maximum_local_curvature_nnz"] for adapter,_rows in self._finite_distributing_bindings)
        if self._definition.footprints:
            details["nodal_rigid_footprints"] = tuple(
                dict(group.descriptor(), point_indices=indices)
                for group, indices in self._footprint_bindings)
            details["footprint_retained_scale_policy"] = retained_scale_policy_descriptor()
            details["footprint_chart_admission_policy"] = chart_admission_policy_descriptor(self._definition.footprint_chart_policy)
        if self._footprint_stable_s3:
            details["footprint_finite_numerical_policy"] = stable_s3_policy_descriptor()
        if self._modal_only:
            details["modal_boundary_policy"] = self._definition.modal_boundary_policy
        from ._stiffened_shell_affine_endpoint import is_native_point_material
        from ._stiffened_shell_affine_endpoint import is_native_material_bl5_arc
        if self._bl5_shell_material or self._curved_reference or is_native_point_material(self) or is_native_material_bl5_arc(self):
            details["arc_metric_node_order"] = (tuple(("shell",int(i)) for i in self._shell_ids)
                +tuple(("beam",int(i)) for i in self._beam_ids))
        if self._coordinate_support:details['shell_support_policy']=NATIVE_SHELL_COORDINATE_SUPPORT_V1
        return details

    @property
    def reference_modal_capabilities(self):
        """Explicit experimental descriptor route; legacy capability schema stays fixed."""
        if self._coordinate_support:raise ValueError('coordinate shell supports exclude spectral capabilities')
        from ._stiffened_shell_ge_reference_modal import capabilities
        with self._operation_lock:
            return capabilities(self)

    @property
    def point_buckling_capabilities(self):
        """Experimental accepted-state route; reference modal schema is unchanged."""
        if self._coordinate_support:raise ValueError('coordinate shell supports exclude spectral capabilities')
        from ._stiffened_shell_ge_point_buckling import capabilities
        with self._operation_lock:
            return capabilities(self)

    @property
    def mixed_reference_modal_capabilities(self):
        """Distinct typed virgin flat Q4/S3/native POINT/BL5 modal extension."""
        if self._coordinate_support:raise ValueError('coordinate shell supports exclude spectral capabilities')
        from ._stiffened_shell_ge_reference_modal import mixed_capabilities
        with self._operation_lock:
            return mixed_capabilities(self)

    @property
    def line_buckling_capabilities(self):
        """Accepted conservative BL5 route, distinct from point-only admission."""
        if self._coordinate_support:raise ValueError('coordinate shell supports exclude spectral capabilities')
        from ._stiffened_shell_ge_point_buckling import capabilities
        with self._operation_lock:
            return capabilities(self, lines=True)

    @property
    def mixed_buckling_capabilities(self):
        """Elastic Q4/S3 matching-point native-family accepted-state route."""
        if self._coordinate_support:raise ValueError('coordinate shell supports exclude spectral capabilities')
        from ._stiffened_shell_ge_point_buckling import capabilities
        with self._operation_lock:
            return capabilities(self, mixed=True)

    @property
    def native_line_buckling_capabilities(self):
        """Explicit elastic physical-fibre BL5 accepted-state descriptor."""
        if self._coordinate_support:raise ValueError('coordinate shell supports exclude spectral capabilities')
        from ._stiffened_shell_ge_point_buckling import capabilities
        with self._operation_lock:
            return capabilities(self, lines=True, mixed=True, native_lines=True)

    @_state_writer
    def buckling(self, request, *, cancellation_token=None):
        """Typed point-only, BL5 or mixed-shell/native accepted-state prestress factors."""
        from ._stiffened_shell_f06_increment_arithmetic import POLICY as COMPENSATED_POLICY
        if any(getattr(e.operator,'arithmetic_policy',None)==COMPENSATED_POLICY for e,_,_ in self._beam_parts):
            raise ValueError('durable compensated native spectra are not implemented')
        if self._coordinate_support:raise ValueError("coordinate shell supports exclude spectral operations")
        if self._bl5_native_material:
            raise ValueError("native material BL5 spectra are not admitted")
        from ._stiffened_shell_ge_point_buckling import solve
        return solve(self, request, cancellation_token=cancellation_token)

    @_state_writer
    def modes(self, request, *, cancellation_token=None):
        """Supported virgin Q4/native-GE point modes with explicit physical mass."""
        from ._stiffened_shell_f06_increment_arithmetic import POLICY as COMPENSATED_POLICY
        if any(getattr(e.operator,'arithmetic_policy',None)==COMPENSATED_POLICY for e,_,_ in self._beam_parts):
            raise ValueError('durable compensated native spectra are not implemented')
        if self._coordinate_support:raise ValueError("coordinate shell supports exclude spectral operations")
        if self._bl5_native_material:
            raise ValueError("native material BL5 spectra are not admitted")
        self._require_distributing_reference_only("spectral operations")
        self._require_finite_distributing_scope("spectral operations")
        if self._curved_reference:raise ValueError("curved reference spectra are not admitted")
        from ._stiffened_shell_ge_reference_modal import solve
        return solve(self, request, cancellation_token=cancellation_token)

    def _require_curved_load(self, load):
        if not self._curved_reference or load is None:return
        nodal=self._shell_count+6*len(self._beam_ids)
        if np.any(load[nodal:]) or np.any(load[:nodal].reshape(-1,6)[:,3:]):
            raise ValueError("curved reference requires nodal translational load coordinates")

    @_state_writer
    def solve_linear(self, actions: tuple[NodalAction, ...], *,
                     cancellation_token=None) -> LinearResult:
        self._require_open()
        if self._coordinate_support:raise ValueError("coordinate shell supports require native fixed-force finite solve")
        if self._bl5_native_material and (type(actions) is not tuple or
                any(type(a) is not NodalAction or any(a.wrench[3:]) for a in actions)):
            raise ValueError("native material BL5 admits nodal dead translations only")
        self._require_finite_distributing_scope("reference-linear solve; select a reference-only record instead")
        if self._distributing_reference and (type(actions) is not tuple or any(type(a) is not NodalAction for a in actions)):
            raise ValueError("distributing reference footprints require a typed tuple of nodal dead actions; load wrappers are unsupported")
        if self._curved_reference and (type(actions) is not tuple or any(
                type(a) is not NodalAction or any(a.wrench[3:]) for a in actions)):
            raise ValueError("curved reference admits nodal dead translations only")
        self._require_stateful_scope()
        if self._finite_driver is not None:
            raise RuntimeError("finite and reference-linear states cannot be mixed")
        if self._unsupported_modes:
            raise ValueError(self._unsupported_modes[0])
        cancellation_safe_point(cancellation_token, "stiffened-shell.linear.start")
        from ._stiffened_shell_ge_body_load import GEBodyLoad
        from ._stiffened_shell_ge_distributed_load import GEBeamDistributedLoad
        from ._stiffened_shell_ge_distributed_programme import GEBeamDistributedProgramme
        if type(actions) is GEBeamDistributedProgramme:
            raise ValueError('distributed programmes require the finite solve operation')
        from ._stiffened_shell_ge_pressure_load import GEShellPressureLoad
        body = None
        if type(actions) in (GEBodyLoad, GEBeamDistributedLoad, GEShellPressureLoad):
            cached = getattr(self, "_external_context", None)
            body = (cached if cached is not None and cached.request.identity == actions.identity
                    else actions.compile(self))
        load = (np.array(body.reference_gradient, copy=True) if body is not None else
                np.zeros(self._K.shape[0]))
        for action in (() if body is not None else actions):
            if type(action) is not NodalAction or action.domain not in ("shell", "beam"):
                raise ValueError("declared shell or beam nodal action required")
            index = self._shell_index if action.domain == "shell" else self._beam_index
            if action.node not in index:
                raise ValueError("action references unknown physical node")
            if self._distributing_reference:
                values = np.asarray(action.wrench)
                if (type(action.node) is not int or values.shape != (6,) or
                        values.dtype.kind not in "fiu" or not np.isfinite(values).all()):
                    raise ValueError("explicit model node and six finite real nodal dead action components required")
            wrench = _owned(action.wrench, (6,))
            offset = 0 if action.domain == "shell" else self._shell_count
            load[offset+6*index[action.node]:offset+6*index[action.node]+6] += wrench
        if self._distributing_reference and not np.isfinite(load).all():
            raise ValueError("representable distributing reference nodal dead load required")
        rhs = np.r_[load[self._free], np.zeros(self._C.shape[0])]
        if self._linear_factor is None:
            self._linear_factor = spla.splu(self._A)
            self._linear_factorizations += 1
        solution = self._linear_factor.solve(rhs)
        cancellation_safe_point(cancellation_token, "stiffened-shell.linear.solved")
        from ._b2_physical_linear_batch_dev import (
            B2NumericalSolveFailure, _accurate_sparse_residual,
        )

        def refine(value):
            if not np.isfinite(value).all():
                raise RuntimeError("finite mixed linear solution required")
            try:
                defect = _accurate_sparse_residual(
                    self._A, value[:, None], rhs[:, None])[:, 0]
            except B2NumericalSolveFailure as exc:
                raise RuntimeError(
                    "mixed reference linear residual is not representable") from exc
            correction = self._linear_factor.solve(-defect)
            self._linear_refinements += 1
            cancellation_safe_point(cancellation_token,
                                    "stiffened-shell.linear.refined")
            return value+correction

        # A rounded sparse residual can pass the physical gates while losing
        # retained forces or multipliers. On fine mixed grids the first accurate
        # correction can itself lose retained-force accuracy even when physical
        # equilibrium passes. Two bounded corrections use this owner's unchanged
        # matrix and cached factor, not a comparison oracle or altered tangent.
        solution = refine(solution)
        solution = refine(solution)
        for attempt in range(2):
            if not np.isfinite(solution).all():
                raise RuntimeError("finite mixed linear solution required")
            q = np.zeros(self._K.shape[0])
            q[self._free] = solution[:len(self._free)]
            multipliers = solution[len(self._free):]
            equilibrium = self._K@q + self._C.T@multipliers-load
            gap = self._C@q
            if getattr(body, 'pressure_vector_equilibrium', False):
                from ._stiffened_shell_ge_finite_experimental import _pressure_vector_equilibrium
                relative = _pressure_vector_equilibrium(
                    equilibrium, load, self._free, self._shell_count,
                    self._coordinate_scale)
            else:
                relative = _linear_equilibrium(
                    equilibrium[self._free], load[self._free],
                    self._coordinate_scale[self._free])
            constraint = _scaled_constraint_gap(gap, self._constraint_scale)
            if np.isfinite(relative) and relative <= 1e-8 and constraint <= 1e-9:
                break
            if attempt == 0:
                solution = refine(solution)
        if not np.isfinite(relative) or relative > 1e-8 or constraint > 1e-9:
            raise RuntimeError("mixed reference linear equilibrium failed")
        cancellation_safe_point(cancellation_token, "stiffened-shell.linear.commit")
        support = np.zeros_like(equilibrium)
        support[self._fixed] = equilibrium[self._fixed]
        point_wrenches = reference_point_records(self, q, multipliers)
        distributing_wrenches = distributing_reference_records(self, q, multipliers, self._generation+1)
        cancellation_safe_point(cancellation_token, "stiffened-shell.linear.publish")
        with self._commit_lock:
            self._generation += 1
            result = LinearResult("completed", _owned(q[:self._shell_count]),
                                  _owned(q[self._shell_count:]),
                                  _owned(multipliers), _owned(support),
                                  relative, constraint, self._generation,
                                  point_wrenches=point_wrenches,
                                  distributing_footprint_wrenches=distributing_wrenches,
                                  **({} if body is None else {body.result_field: body.reference_recovery}))
            self._external_context = body
            self._accepted = (result, _owned(load))
        return result

    @_state_writer
    def solve(self, actions: tuple[NodalAction, ...],
              targets: tuple[float, ...] = (1.,), *, cancellation_token=None,
              max_iterations: int = 12, max_backtracks: int = 8,
              increment_control=None, on_accepted=None, displacement_control=None):
        """Advance finite trials; optionally observe each committed increment.

        ``on_accepted(result)`` receives a detached result after state and
        controller publication. Recovery/checkpoint reads are allowed; reentrant
        mutation is rejected. Observer errors propagate with the accepted prefix
        intact. The observer is ephemeral and is not stored in checkpoints.
        """
        from ._stiffened_shell_f06_increment_arithmetic import POLICY as COMPENSATED_POLICY
        if any(getattr(e.operator,'arithmetic_policy',None)==COMPENSATED_POLICY for e,_,_ in self._beam_parts):
            from ._stiffened_shell_ge_finite_experimental import compensated_adaptive_load_admitted
            from ._stiffened_shell_displacement_control import DisplacementControl
            from ._stiffened_shell_affine_endpoint import is_paired_displacement
            paired_control=(type(displacement_control) is DisplacementControl and
                is_paired_displacement(self) and displacement_control.force_seed_sha256 is not None and
                displacement_control.domain=="shell" and displacement_control.trial_policy is None)
            if ((displacement_control is not None and not paired_control) or (increment_control is not None and
                    not paired_control and not compensated_adaptive_load_admitted(self, actions))):
                raise ValueError('durable compensated adaptive control requires an admitted staged-fibre or nodal-mixed point owner')
        self._require_open()
        self._require_distributing_reference_only("finite solve and its controllers")
        if self._coordinate_support and (type(actions) is not tuple or
                any(type(a) is not NodalAction or any(a.wrench[3:]) for a in actions) or
                increment_control is not None or displacement_control is not None or
                getattr(self,'_arc_snapshot',None) is not None or
                getattr(self,'_external_context',None) is not None or getattr(self,'_staged_context',None) is not None):
            raise ValueError("coordinate shell supports require fixed-force nodal translations; controllers/wrappers excluded")
        if self._bl5_native_material and (
                type(actions) is not tuple or any(type(a) is not NodalAction or any(a.wrench[3:]) for a in actions)):
            raise ValueError("native material BL5 requires nodal dead translations")
        if self._finite_distributing:
            if self._distributing_cyclic and increment_control is not None:
                raise ValueError("cyclic distributing material histories admit fixed increments only")
            if displacement_control is not None and increment_control is not None:
                raise ValueError("finite distributing displacement excludes adaptive control")
            if type(actions) is not tuple or any(type(a) is not NodalAction for a in actions):
                raise ValueError("finite distributing footprints admit nodal translation forces only")
        from ._stiffened_shell_staged_load_experimental import StagedDeadLoadProgramme
        curved_staged = self._curved_material and type(actions) is StagedDeadLoadProgramme
        if curved_staged and displacement_control is not None:
            raise ValueError("staged programme positions require force-control stepping")
        if self._curved_reference and not curved_staged and (type(actions) is not tuple or any(
                type(a) is not NodalAction or any(a.wrench[3:]) for a in actions)):
            raise ValueError("curved reference admits nodal dead translations only")
        self._require_stateful_scope()
        if on_accepted is not None and not callable(on_accepted):
            raise TypeError("on_accepted must be callable or None")
        if getattr(self,"_arc_snapshot",None) is not None:
            raise ValueError("arc history requires its dedicated continuation controller")
        if displacement_control is not None:
            if self._bl5_native_material and increment_control is not None:
                raise ValueError("native material BL5 displacement excludes adaptive control")
            if self._unsupported_modes:
                raise ValueError(self._unsupported_modes[0])
            if increment_control is not None:
                from ._stiffened_shell_displacement_adaptive import solve as adaptive_solve
                return adaptive_solve(self,actions,targets,displacement_control,increment_control,
                    max_iterations=max_iterations,max_backtracks=max_backtracks,
                    cancellation_token=cancellation_token,on_accepted=on_accepted)
            from ._stiffened_shell_displacement_control import solve
            return solve(self,actions,targets,displacement_control,
                max_iterations=max_iterations,max_backtracks=max_backtracks,
                cancellation_token=cancellation_token,on_accepted=on_accepted)
        if getattr(self,"_displacement_snapshot",None) is not None or getattr(self,"_displacement_adaptive",None) is not None:
            raise ValueError("displacement history requires explicit displacement control")
        if self._unsupported_modes:
            raise ValueError(self._unsupported_modes[0])
        if self._line_samples and not self._finite_bl5:
            raise ValueError("experimental weak line admits reference-linear analysis only")
        if self._accepted is not None:
            raise RuntimeError("finite and reference-linear states cannot be mixed")
        if self._finite_driver is None:
            from ._stiffened_shell_ge_finite_experimental import FinitePointDriver
            driver = FinitePointDriver(self)
        else:
            driver = self._finite_driver
        try:
            return driver.solve(
                actions, targets, cancellation_token=cancellation_token,
                max_iterations=max_iterations, max_backtracks=max_backtracks,
                increment_control=increment_control, on_accepted=on_accepted)
        finally:
            if self._finite_driver is None and driver._accepted is not None:
                self._finite_driver = driver

    @_accepted_snapshot
    def recover(self, *, selection=None):
        self._require_open()
        if self._finite_driver is not None:
            result = self._finite_driver.recover()
        else:
            if self._accepted is None:
                raise RuntimeError("no accepted mixed linear result")
            result = self._accepted[0]
        if type(selection) is DistributingFootprintSelection:
            return selected_distributing_records(result, selection)
        if self._distributing_reference and selection is not None:
            raise ValueError("typed reference distributing footprint selection required")
        from ._stiffened_shell_selected_recovery import GERecoverySelection, recover
        if type(selection) is GERecoverySelection:
            return recover(self, result, selection)
        return result if selection is None else selected_wrenches(self, result, selection)

    @_state_writer
    def rebase_rotations(self, expected_generation: int, *, cancellation_token=None,
                         preserve_supported_charts=False):
        """Issue an experimental physical shell-anchor reset at the same load."""
        self._require_open()
        if type(preserve_supported_charts) is not bool:
            raise ValueError("preserve_supported_charts must be a strict boolean")
        if preserve_supported_charts:
            if (self._definition.shell.q4_elements and
                    self._definition.q4_pullback_policy != Q4_ALGORITHMIC_PULLBACK_V1):
                raise ValueError("selective chart rebase requires explicit algorithmic Q4 pullback policy")
            driver=self._finite_driver
            if (self._definition.lines or self._definition.footprints or
                    getattr(self,"_arc_snapshot",None) is not None or
                    getattr(self,"_displacement_snapshot",None) is not None or
                    getattr(self,"_displacement_adaptive",None) is not None or
                    getattr(self,"_external_context",None) is not None or
                    getattr(self,"_staged_context",None) is not None or
                    (driver is not None and (driver._adaptive_state is not None or
                        driver._programme is not None or driver._external is not None))):
                raise ValueError("selective chart rebase requires fixed force-controlled native point scope")
        if self._bl5_native_material:
            if type(expected_generation) is not int or expected_generation != self._generation:
                raise ValueError("stale or invalid accepted rebase predecessor")
            raise ValueError("native material BL5 chart rebase is not admitted")
        self._require_distributing_reference_only("chart rebase")
        self._require_finite_distributing_scope("chart rebase")
        if getattr(self,"_arc_snapshot",None) is not None:
            raise ValueError("arc chart rebasing is not yet admitted")
        if getattr(self,"_displacement_snapshot",None) is not None or getattr(self,"_displacement_adaptive",None) is not None:
            from ._stiffened_shell_displacement_control import rebase
            return rebase(self,expected_generation,cancellation_token=cancellation_token)
        self._require_stateful_scope()
        if self._definition.footprints and not self._footprint_increment_chart:
            raise ValueError("experimental nodal footprint chart rebase is unsupported: continuation parity not established")
        if self._finite_driver is None:
            raise RuntimeError("shell rebase requires an accepted finite point state")
        return self._finite_driver.rebase_rotations(
            expected_generation, cancellation_token=cancellation_token,
            preserve_supported_charts=preserve_supported_charts)

    @_accepted_snapshot
    def checkpoint(self) -> bytes:
        self._require_open()
        if self._finite_distributing and any(getattr(self, name, None) is not None for name in
                ("_arc_snapshot", "_displacement_adaptive", "_external_context")):
            raise ValueError("finite distributing checkpoint excludes wrapper/controller state")
        if getattr(self,"_arc_snapshot",None) is not None:
            return self._arc_snapshot[0]
        if getattr(self,"_displacement_adaptive",None) is not None:
            return self._displacement_adaptive[1]
        if getattr(self,"_displacement_snapshot",None) is not None:
            return self._displacement_snapshot[0]
        if self._finite_driver is not None:
            return self._finite_driver.checkpoint()
        if self._accepted is None:
            raise RuntimeError("no accepted mixed linear result")
        body = getattr(self, "_external_context", None)
        if body is not None:
            prepared = copy(self)
            prepared._external_context = None
            prepared._identity = body.augmented_model_identity
            return body.checkpoint(prepared.checkpoint())
        result, load = self._accepted
        arrays = (("q", np.r_[result.shell_displacement,
                               result.beam_coordinates]),
                  ("multipliers", result.multipliers), ("load", load))
        if sum(value.size for _name, value in arrays) > MAX_JSON_CHECKPOINT_VALUES:
            return _encode_binary_checkpoint(
                "linear", self._identity, self._generation, arrays)
        payload = {"schema": CHECKPOINT, "model": self._identity,
                   "generation": self._generation,
                   "q": arrays[0][1].tolist(),
                   "multipliers": arrays[1][1].tolist(),
                   "load": arrays[2][1].tolist()}
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("ascii")
        if len(raw) > MAX_JSON_CHECKPOINT_BYTES:
            return _encode_binary_checkpoint(
                "linear", self._identity, self._generation, arrays)
        return raw

    @_state_writer
    def continue_arc_length(self, actions, control, *, steps=1,
                            cancellation_token=None, on_accepted=None):
        self._require_open()
        from ._stiffened_shell_affine_endpoint import is_paired_displacement
        if is_paired_displacement(self):
            raise ValueError("paired mixed Q4-J2 arc control is not admitted")
        from ._stiffened_shell_affine_endpoint import is_native_material_bl5_arc
        if self._bl5_native_material and not is_native_material_bl5_arc(self):
            raise ValueError("native material BL5 arc requires its explicit admitted material domain")
        self._require_distributing_reference_only("arc-length continuation")
        self._require_finite_distributing_scope("arc-length continuation")
        self._require_stateful_scope()
        if self._coordinate_support:raise ValueError('coordinate shell supports exclude arc control')
        from ._stiffened_shell_arc_control import solve
        return solve(self,actions,control,steps=steps,cancellation_token=cancellation_token,
                     on_accepted=on_accepted)

    @_state_writer
    @_accepted_snapshot
    def restore(self, raw: bytes, *, legacy_compensated_base=False, cancellation_token=None) -> LinearResult:
        self._require_open()
        self._require_stateful_scope()
        if self._unsupported_modes:
            raise ValueError(self._unsupported_modes[0])
        if self._finite_driver is not None:
            raise RuntimeError("restore requires a fresh experimental GE analysis")
        if self._accepted is not None:
            raise RuntimeError("restore requires a fresh experimental GE analysis")
        if type(raw) is not bytes:
            raise ValueError("bounded mixed linear checkpoint required")
        # The selected polishing candidate binds one effective identity at
        # every framed restore layer: its tagged bytes restore only with the
        # policy selected, and legacy bytes restore only without it. Every
        # unselected analysis keeps its exact legacy identity, so OFF bytes
        # and legacy owners built before the selector existed (missing
        # selector means False) restore exactly as before.
        from ._stiffened_shell_ge_finite_experimental import admissible_polish_identity
        polish_selected = getattr(self, "_admissible_polish_candidate", False) is True
        effective_identity = (admissible_polish_identity(self._identity)
                              if polish_selected else self._identity)
        from ._stiffened_shell_displacement_adaptive import PAIRED_MAGIC as PAIRED_ADAPTIVE_MAGIC,PAIRED_INTERPOLATION_MAGIC
        if raw.startswith((PAIRED_ADAPTIVE_MAGIC,PAIRED_INTERPOLATION_MAGIC)):
            if legacy_compensated_base is not False:
                raise ValueError("paired adaptive checkpoint excludes legacy base import")
            from ._stiffened_shell_displacement_adaptive import restore_paired
            return restore_paired(self,raw,cancellation_token=cancellation_token)
        if cancellation_token is not None:
            raise ValueError("restore cancellation is implemented only for paired adaptive checkpoints")
        from ._stiffened_shell_displacement_checkpoint import PAIRED_MAGIC
        if raw.startswith(PAIRED_MAGIC):
            from ._stiffened_shell_affine_endpoint import is_paired_displacement
            if not is_paired_displacement(self) or legacy_compensated_base is not False:
                raise ValueError("paired displacement checkpoint requires exact selected mixed Q4-J2 owner")
            from ._stiffened_shell_displacement_control import publish
            return publish(self,raw,expected_driver=None,expected_generation=self._generation)
        from ._stiffened_shell_f06_increment_arithmetic import POLICY as COMPENSATED_POLICY
        from ._stiffened_shell_ge_checkpoint_binary import COMPENSATED_MAGIC,decode_compensated
        from ._stiffened_shell_staged_load_experimental import MAGIC as STAGED_MAGIC
        selected=any(getattr(e.operator,'arithmetic_policy',None)==COMPENSATED_POLICY for e,_,_ in self._beam_parts)
        corrections=None
        if raw.startswith(COMPENSATED_MAGIC):
            if not selected: raise ValueError('compensated checkpoint requires selected arithmetic policy')
            shapes=(('q_low',(self._K.shape[0],)),('frames_low',(len(self._beam_ids),3,3)),
                    ('cells_low',(len(self._beam_parts),2,3,3)),('resultants_low',(len(self._beam_parts),18)))
            raw,corrections=decode_compensated(raw,effective_identity,shapes)
        elif selected and not raw.startswith(STAGED_MAGIC):
            if legacy_compensated_base is not True:
                raise ValueError('legacy checkpoint lacks durable corrections; explicit declared binary base import required')
            # A verified legacy endpoint is imported as a NEW declared physical
            # binary base. No discarded correction or loading history is inferred.
            from ._stiffened_shell_ge_finite_experimental import FinitePointDriver
            driver=FinitePointDriver(self)  # Admit and prepare the selected owner.
            from ._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement
            legacy_beams=[]
            for definition in self._definition.beams:
                element,inertia=definition.instantiate()
                core=element.operator
                legacy=NativeGeneralizedStaticElement(element.element_id,element.node_ids,
                    core.reference,core.section,order=core.order)
                legacy_beams.append(NativeBeamDefinition.capture(legacy,inertia))
            legacy_owner=ExperimentalGEAnalysis(replace(self._definition,beams=tuple(legacy_beams)))
            try:
                result=legacy_owner.restore(raw)
                if legacy_owner._finite_driver is None:
                    raise ValueError('declared compensated base requires finite native state')
                legacy_driver=legacy_owner._finite_driver
                # Only durable accepted fields migrate. Prepared operators,
                # geometry caches and trial captures belong to the NEW owner.
                for name in ('_state','_shell_origins','_beam_origins','_shell_predecessors',
                        '_beam_predecessors','_shell_predecessor_q','_shell_predecessor_anchors',
                        '_pattern','_load_factor','_accepted_factors','_accepted',
                        '_curved_material_predecessors'):
                    setattr(driver,name,getattr(legacy_driver,name))
                driver._state=driver._zero_compensation(driver._state)
                residual,_,relative,gap,_,_=driver._assemble(driver._state,driver._load_factor,
                    driver._shell_predecessors,driver._beam_predecessors)
                if not np.isfinite(relative) or relative>1e-8 or gap>1e-9:
                    raise ValueError('mechanically inconsistent declared compensated base')
                self._generation=result.generation
                self._finite_driver=driver
                return result
            finally:
                legacy_owner.close()
        elif legacy_compensated_base is not False:
            raise ValueError('declared compensated base requires selected arithmetic policy')
        from ._stiffened_shell_displacement_checkpoint import MAGIC as DISPLACEMENT_MAGIC
        if self._finite_distributing and not raw.startswith((BINARY_CHECKPOINT_MAGIC, INCREMENT_CONTROL_MAGIC, DISPLACEMENT_MAGIC)):
            raise ValueError("finite distributing checkpoints require dedicated binary state or adaptive envelope")
        self._require_distributing_linear_checkpoint(raw)
        from ._stiffened_shell_arc_checkpoint import MAGIC as ARC_MAGIC
        if raw.startswith(ARC_MAGIC):
            from ._stiffened_shell_arc_publication import publish
            return publish(self,raw,expected_driver=None,expected_generation=self._generation,restoring=True)
        from ._stiffened_shell_displacement_adaptive import MAGICS as DISPLACEMENT_ADAPTIVE_MAGICS
        if raw.startswith(DISPLACEMENT_ADAPTIVE_MAGICS):
            if self._bl5_native_material:
                raise ValueError("native material BL5 displacement excludes adaptive checkpoints")
            if self._bl5_native_material:
                raise ValueError("native material BL5 displacement excludes adaptive checkpoints")
            from ._stiffened_shell_displacement_adaptive import decode,validate_progress
            context,inner=decode(self,raw)
            prepared=copy(self);prepared._commit_lock=RLock();prepared._writing=False
            prepared._displacement_adaptive=None
            result=prepared.restore(inner)
            validate_progress(prepared,context)
            if getattr(prepared,"_displacement_snapshot",None) is not None:
                result=replace(result,status="completed" if context.state.completed else "accepted_prefix",
                               increment_attempts=context.state.attempts)
                prepared._finite_driver._accepted=result
            with self._commit_lock:
                self._finite_driver=prepared._finite_driver
                self._finite_driver.owner=self
                self._generation=prepared._generation
                self._displacement_snapshot=getattr(prepared,"_displacement_snapshot",None)
                self._displacement_adaptive=(context,raw)
            return result
        from ._stiffened_shell_displacement_checkpoint import MAGIC as DISPLACEMENT_MAGIC
        if raw.startswith(DISPLACEMENT_MAGIC):
            from ._stiffened_shell_displacement_control import publish
            return publish(self,raw,expected_driver=None,expected_generation=self._generation)
        from ._stiffened_shell_ge_body_load import MAGIC as BODY_MAGIC
        from ._stiffened_shell_ge_distributed_load import MAGIC as DISTRIBUTED_MAGIC
        from ._stiffened_shell_ge_pressure_load import MAGIC as PRESSURE_MAGIC
        from ._stiffened_shell_ge_distributed_programme import MAGIC as DISTRIBUTED_PROGRAMME_MAGIC
        if raw.startswith((BODY_MAGIC, DISTRIBUTED_MAGIC, PRESSURE_MAGIC, DISTRIBUTED_PROGRAMME_MAGIC)):
            if self._bl5_native_material:
                raise ValueError("native material BL5 force-controlled scope excludes external load checkpoints")
            if self._curved_reference:raise ValueError("curved reference excludes external load checkpoints")
            if getattr(self, "_external_context", None) is not None or getattr(self, "_staged_context", None) is not None:
                raise ValueError("nested external-load checkpoint is unsupported")
            if raw.startswith(BODY_MAGIC):
                from ._stiffened_shell_ge_body_load import decode_checkpoint
            elif raw.startswith(DISTRIBUTED_MAGIC):
                from ._stiffened_shell_ge_distributed_load import decode_checkpoint
            elif raw.startswith(DISTRIBUTED_PROGRAMME_MAGIC):
                from ._stiffened_shell_ge_distributed_programme import decode_checkpoint
            else:
                from ._stiffened_shell_ge_pressure_load import decode_checkpoint
            request, inner = decode_checkpoint(raw)
            compiled = request.compile(self)
            prepared = copy(self)
            prepared._identity = compiled.augmented_model_identity
            prepared._external_context = compiled
            prepared._commit_lock = RLock()
            prepared._writing = False
            result = prepared.restore(inner)
            driver = prepared._finite_driver
            if driver is None:
                if hasattr(compiled, 'evaluate_at'):
                    raise ValueError('distributed programme requires a finite checkpoint')
                if not np.array_equal(prepared._accepted[1], compiled.reference_gradient):
                    raise ValueError("checkpoint load disagrees with its declared external request")
                result = replace(result, **{compiled.result_field: compiled.reference_recovery})
                prepared._accepted = (result, prepared._accepted[1])
            elif hasattr(compiled, 'evaluate_at') and driver._adaptive_state is not None and not driver._adaptive_state.completed:
                controller = driver._adaptive_state
                compiled.validate_targets(driver._load_factor, controller.targets[controller.cursor:])
            checkpoint = prepared.checkpoint()
            if driver is not None:
                driver.owner = self
                driver._adaptive_checkpoint = checkpoint
            with self._commit_lock:
                self._external_context = compiled
                self._finite_driver = driver
                self._accepted = prepared._accepted
                self._generation = result.generation
            return result
        from ._stiffened_shell_staged_load_experimental import (
            MAGIC as STAGED_MAGIC, decode_checkpoint,
        )
        if raw.startswith(STAGED_MAGIC):
            if corrections is not None:
                raise ValueError('staged load envelope must be outermost')
            if self._bl5_native_material:
                raise ValueError("native material BL5 force-controlled scope excludes staged load checkpoints")
            if self._curved_reference and not self._curved_material:
                raise ValueError("curved reference excludes staged load checkpoints")
            if ((self._line_samples and not (self._finite_bl5 and
                    self._definition.q4_pullback_policy == Q4_ALGORITHMIC_PULLBACK_V1)) or
                    getattr(self, "_staged_context", None) is not None or
                    getattr(self, "_external_context", None) is not None):
                raise ValueError("staged checkpoint requires a fresh point or admitted elastic BL5 owner")
            from ._stiffened_shell_ge_finite_experimental import FinitePointDriver
            programme, position, inner = decode_checkpoint(raw)
            compiled = programme.compile(self, FinitePointDriver(self)._load)
            # Validate in a private view; neither model identity nor accepted
            # state on the live owner changes before every check passes.
            prepared = copy(self)
            prepared._identity = compiled.augmented_model_identity
            prepared._staged_context = compiled
            prepared._commit_lock = RLock()
            prepared._writing = False
            # Staged-only auto restore inference of the private polishing
            # candidate: after the staged decode/compile validated the
            # envelope, a fresh owner with no explicit selection infers the
            # polished mode from the inner checkpoint's tagged identity, on
            # this private copy only. Raw legacy bytes keep the default
            # route; explicit True/False stays binding; tagged bytes of a
            # different owner or a non-staged tagged checkpoint keep their
            # existing explicit refusal, and a refused checkpoint never
            # mutates the live owner.
            from ._stiffened_shell_ge_finite_experimental import (
                _explicit_admissible_polish, _probe_checkpoint_model,
                supports_native_chart_polish)
            if (_explicit_admissible_polish(self) is None and not selected and
                    supports_native_chart_polish(prepared) and
                    _probe_checkpoint_model(inner) ==
                        admissible_polish_identity(compiled.augmented_model_identity)):
                prepared._admissible_polish_candidate = True
            result = prepared.restore(inner)
            driver = prepared._finite_driver
            if driver is None or result.load_position != position:
                raise ValueError("staged checkpoint progress disagrees with accepted state")
            if driver._adaptive_state is not None and not driver._adaptive_state.completed:
                controller = driver._adaptive_state
                compiled.validate_targets(driver._load_factor,
                                          controller.targets[controller.cursor:])
            # Prepare the durable representation before the single publish.
            checkpoint = driver.checkpoint()
            driver.owner = self
            with self._commit_lock:
                self._finite_driver = driver
                self._generation = result.generation
                driver._adaptive_checkpoint = checkpoint
            return result
        load_key = ("applied_dead_load" if getattr(self, "_staged_context", None)
                    is not None else "pattern")
        adaptive = None
        if raw.startswith(INCREMENT_CONTROL_MAGIC):
            if self._distributing_cyclic:
                raise ValueError("cyclic distributing material checkpoints exclude adaptive envelopes")
            adaptive, raw = decode_control(raw, effective_identity)
            if selected:
                from ._stiffened_shell_ge_finite_experimental import supports_paired_fibre_adaptive, supports_mixed_material_adaptive
                staged = getattr(self, '_staged_context', None) is not None
                if (corrections is None or not ((supports_paired_fibre_adaptive(self) and staged) or
                        (supports_mixed_material_adaptive(self) and not staged))):
                    raise ValueError('paired adaptive checkpoint requires its exact admitted load/material context')
            if self._finite_distributing and not raw.startswith(BINARY_CHECKPOINT_MAGIC):
                raise ValueError("adaptive distributing checkpoint requires its dedicated binary inner state")
        chart = None
        stable_q4 = False
        if self._has_physical_fibre and raw.startswith(S3_BINARY_CHECKPOINT_MAGIC):
            raise ValueError("physical-fibre mixed owner admits reference-linear checkpoints only")
        if raw.startswith(CHART_CHECKPOINT_MAGIC):
            if self._bl5_native_material:
                raise ValueError("native material BL5 force-controlled scope excludes chart checkpoints")
            if self._definition.footprints:
                if not self._footprint_increment_chart:
                    raise ValueError("experimental nodal footprint chart checkpoint restore is unsupported: continuation parity not established")
                require_rotation_triples(self._fixed, len(self._shell_ids))
            if self._line_samples and not self._finite_bl5:
                raise ValueError("owned shell charts require a supported finite model")
            event_generation, anchors, witness, raw = _decode_chart_checkpoint(
                raw, effective_identity, len(self._shell_ids))
            chart = (event_generation, anchors, witness)
        if raw.startswith(BINARY_CHECKPOINT_MAGIC):
            def expected_shapes(kind, generation):
                if self._finite_distributing and kind != self._distributing_checkpoint_kind:
                    raise ValueError("finite distributing checkpoint requires its exact mechanics kind")
                if kind in ("finite_distributing",) + FINITE_DISTRIBUTING_MATERIAL_KINDS and not self._finite_distributing:
                    raise ValueError("finite distributing checkpoint requires matching attachment records")
                if self._distributing_reference and kind != "linear":
                    raise ValueError("distributing footprints require reference-linear binary checkpoints; finite kinds are unsupported")
                if ("_chart" in kind) != (chart is not None):
                    raise ValueError("numeric shell chart snapshot requires matching outer policy")
                kind = kind.replace("_chart", "")
                kind = kind.removesuffix("_stable_q4")
                if selected:
                    from ._stiffened_shell_beam_history import compensated_kind
                    expected = compensated_kind(self._beam_parts,
                        coupled=self._definition.shell.j2_curve is not None)
                    if kind != expected:
                        raise ValueError('compensated native state requires its exact material schema')
                stable_kind = kind == "finite_native_material_stable"
                body_stable = bool(getattr(self, "_external_context", None) is not None and
                    self._definition.shell.s3_elements and self._definition.shell.j2_curve is None)
                if body_stable and kind not in ("linear", "finite_native_material_stable"):
                    raise ValueError("body S3 checkpoint requires its stable numerical policy")
                if self._footprint_stable_s3 and kind not in ("linear", "finite_native_material_stable"):
                    raise ValueError("nodal footprint S3 checkpoint requires its stable numerical policy")
                if kind in CURVED_POINT_MATERIAL_KINDS and (not self._curved_material or
                        (kind == "finite_curved_points_coupled") != (self._definition.shell.j2_curve is not None)):
                    raise ValueError("curved material predecessor checkpoint requires its matching policy")
                if (kind == "finite_bl5_coupled") != self._bl5_shell_material and kind != "linear":
                    raise ValueError("matching coupled BL5 material checkpoint required")
                if kind == "finite_bl5_material" and not self._bl5_material:
                    raise ValueError("BL5 material checkpoint requires selected material policy")
                if kind == "finite_bl5" and self._bl5_material:
                    raise ValueError("BL5 material checkpoint must retain histories")
                if kind == "finite_bl5" and not self._finite_bl5:
                    raise ValueError("finite BL5 checkpoint requires elastic BL5 line model")
                if self._bl5_native_mixed_shell and kind not in ("linear", "finite_native_material_stable"):
                    raise ValueError("mixed-shell native BL5 checkpoint requires its stable S3 numerical policy")
                if self._line_samples and kind not in (("linear", "finite_native_material", "finite_native_material_stable") if self._bl5_native_mixed_shell else
                        ("linear", "finite_native_material") if self._bl5_native_fibre else
                        ("linear", "finite_bl5", "finite_bl5_material", "finite_bl5_coupled")):
                    raise ValueError("experimental weak line requires its finite BL5 checkpoint identity"
                                     if self._finite_bl5 else
                                     "experimental weak line admits reference-linear checkpoints only")
                if kind == "finite_native_material_stable":
                    if not self._definition.shell.s3_elements:
                        raise ValueError("stable S3 native checkpoint requires S3 elements")
                    kind = "finite_native_material"
                if kind not in ("linear", "finite_native_material", "finite_compensated_fibre_material", "finite_compensated_mixed_material", "finite_compensated_generalized_coupled", "finite_compensated_fibre_coupled", "finite_compensated_mixed_coupled") + FINITE_DISTRIBUTING_NATIVE_KINDS and self._has_physical_fibre:
                    raise ValueError("physical-fibre mixed owner admits reference-linear checkpoints only")
                count = self._K.shape[0]
                constraints = self._C.shape[0]
                if kind == "linear":
                    return (("q", (count,)), ("multipliers", (constraints,)),
                            ("load", (count,)))
                members = len(self._beam_parts)
                base = (("q", (count,)),
                        ("frames", (len(self._beam_ids), 3, 3)),
                        ("cells", (members, 2, 3, 3)),
                        ("resultants", (members, 18)),
                        ("multipliers", (constraints,)),
                        (load_key, (count,)),
                        ("accepted_factors", (generation,)))
                if kind in ('finite_compensated_generalized_material', 'finite_compensated_fibre_material', 'finite_compensated_mixed_material', 'finite_compensated_generalized_coupled', 'finite_compensated_fibre_coupled', 'finite_compensated_mixed_coupled'):
                    from ._stiffened_shell_f06_increment_arithmetic import POLICY as COMPENSATED_POLICY
                    from ._stiffened_shell_affine_endpoint import is_native_point_material
                    from ._stiffened_shell_beam_history import compensated_kind
                    coupled = self._definition.shell.j2_curve is not None
                    if ((coupled and (self._definition.shell.s3_elements or not self._definition.shell.q4_elements)) or
                            kind != compensated_kind(self._beam_parts, coupled=coupled) or
                            not is_native_point_material(self) or not all(
                            getattr(e.operator,'arithmetic_policy',None)==COMPENSATED_POLICY for e,_,_ in self._beam_parts)):
                        raise ValueError('exact compensated native point material domain required')
                    from ._stiffened_shell_beam_history import shapes
                    material = (("shell_plastic", (len(self._shell_parts),12,3)),
                        ("shell_alpha", (len(self._shell_parts),12)),
                        ("shell_origin_plastic", (len(self._shell_parts),12,3)),
                        ("shell_origin_alpha", (len(self._shell_parts),12))) if coupled else ()
                    return base+shapes(self._beam_parts,'origin')+shapes(self._beam_parts,'accepted')+material
                if kind == "finite_native_material":
                    from ._stiffened_shell_ge_finite_experimental import (
                        supports_generalized_stable_s3)
                    if ((not self._has_physical_fibre and not
                         ((self._footprint_stable_s3 or body_stable) and stable_kind) and
                         not (stable_kind and supports_generalized_stable_s3(self))) or
                            self._definition.shell.j2_curve is not None):
                        raise ValueError("native material checkpoint requires elastic-shell fibre model")
                    from ._stiffened_shell_beam_history import shapes
                    return base + shapes(self._beam_parts, "origin") + shapes(self._beam_parts, "accepted")
                if kind in FINITE_DISTRIBUTING_NATIVE_KINDS:
                    if not self._distributing_native or not self._has_physical_fibre:
                        raise ValueError("native distributing checkpoint requires selected fibre policy")
                    from ._stiffened_shell_beam_history import shapes
                    material = base + shapes(self._beam_parts, "origin") + shapes(self._beam_parts, "accepted")
                    if self._distributing_shell_material:
                        shells = len(self._shell_parts)
                        material += (("shell_plastic", (shells, 12, 3)),
                                     ("shell_alpha", (shells, 12)),
                                     ("shell_origin_plastic", (shells, 12, 3)),
                                     ("shell_origin_alpha", (shells, 12)))
                    return material
                if kind in ("finite_material", "finite_shell_material", "finite_bl5_material", "finite_bl5_coupled") + CURVED_POINT_MATERIAL_KINDS + FINITE_DISTRIBUTING_MATERIAL_KINDS:
                    stations = sum(len(part[0].operator.cell.stations)
                                   for part in self._beam_parts)
                    material = base + (("beam_plastic", (stations, 6, 2)),
                                       ("beam_accumulated", (stations, 2)))
                    if kind in ("finite_bl5_material", "finite_bl5_coupled") + CURVED_POINT_MATERIAL_KINDS + FINITE_DISTRIBUTING_MATERIAL_KINDS:
                        material += (("beam_origin_plastic", (stations, 6, 2)),
                                     ("beam_origin_accumulated", (stations, 2)))
                    if kind in ("finite_bl5_coupled", "finite_curved_points_coupled", "finite_distributing_coupled", "finite_distributing_cyclic_coupled"):
                        shells = len(self._shell_parts)
                        material += (("shell_plastic", (shells, 12, 3)),
                                     ("shell_alpha", (shells, 12)),
                                     ("shell_origin_plastic", (shells, 12, 3)),
                                     ("shell_origin_alpha", (shells, 12)))
                    if kind == "finite_shell_material":
                        if self._definition.shell.j2_curve is None:
                            raise ValueError("shell J2 checkpoint requires matching material")
                        material += (("shell_plastic", (len(self._shell_parts), 12, 3)),
                                     ("shell_alpha", (len(self._shell_parts), 12)))
                    return material
                return base

            kind, generation, arrays = _decode_binary_checkpoint(
                raw, effective_identity, expected_shapes)
            kind = kind.replace("_chart", "")
            stable_q4 = kind.endswith("_stable_q4")
            kind = kind.removesuffix("_stable_q4")
            schema = (CHECKPOINT if kind == "linear" else
                      _BINARY_SCHEMAS[kind] if kind in ("finite_compensated_generalized_material", "finite_compensated_fibre_material", "finite_compensated_mixed_material", "finite_compensated_generalized_coupled", "finite_compensated_fibre_coupled", "finite_compensated_mixed_coupled", "finite_distributing", "finite_bl5", "finite_bl5_material", "finite_bl5_coupled") + CURVED_POINT_MATERIAL_KINDS + FINITE_DISTRIBUTING_MATERIAL_KINDS else
                      STABLE_NATIVE_SCHEMA if kind == "finite_native_material_stable" else
                      "B3_GE_E4_PL_STIFFENED_SHELL_EXPERIMENTAL_NATIVE_MATERIAL_V7"
                      if kind == "finite_native_material" else
                      FINITE_MATERIAL_CHECKPOINT if kind == "finite_material" else
                      FINITE_SHELL_MATERIAL_CHECKPOINT if kind == "finite_shell_material"
                      else FINITE_CHECKPOINT)
            value = {"schema": schema, "model": effective_identity,
                     "generation": generation, **arrays}
            if kind in ("finite_compensated_generalized_material", "finite_compensated_fibre_material", "finite_compensated_mixed_material", "finite_compensated_generalized_coupled", "finite_compensated_fibre_coupled", "finite_compensated_mixed_coupled", "finite", "finite_distributing", "finite_bl5", "finite_bl5_material", "finite_bl5_coupled", "finite_material", "finite_shell_material", "finite_native_material", "finite_native_material_stable") + CURVED_POINT_MATERIAL_KINDS + FINITE_DISTRIBUTING_MATERIAL_KINDS:
                value["accepted_factors"] = arrays["accepted_factors"].tolist()
        elif raw.startswith(NATIVE_SHELL_MAGIC):
            if ((self._line_samples and not self._bl5_native_material) or not self._has_physical_fibre or
                    self._definition.shell.j2_curve is None):
                raise ValueError("native shell J2 checkpoint requires matching fibre point model")
            from ._stiffened_shell_ge_finite_experimental import native_shell_checkpoint_shapes
            generation, arrays, states = _decode_s3_binary_checkpoint(
                raw, effective_identity, lambda generation: native_shell_checkpoint_shapes(self, generation),
                2*len(self._definition.shell.s3_elements), anchored=chart is not None,
                native_material=True)
            value = {"schema": NATIVE_SHELL_SCHEMA, "model": effective_identity,
                     "generation": generation, **arrays, "s3_states": states,
                     "accepted_factors": arrays["accepted_factors"].tolist()}
        elif raw.startswith(S3_BINARY_CHECKPOINT_MAGIC):
            if (self._line_samples or self._definition.shell.j2_curve is None or
                    not self._definition.shell.s3_elements):
                raise ValueError("S3 J2 checkpoint requires matching point model")

            def expected_s3_shapes(generation):
                count = self._K.shape[0]
                members = len(self._beam_parts)
                stations = sum(len(part[0].operator.cell.stations)
                               for part in self._beam_parts)
                q4_count = len(self._definition.shell.q4_elements)
                return (("q", (count,)),
                        ("frames", (len(self._beam_ids), 3, 3)),
                        ("cells", (members, 2, 3, 3)),
                        ("resultants", (members, 18)),
                        ("multipliers", (self._C.shape[0],)),
                        (load_key, (count,)),
                        ("accepted_factors", (generation,)),
                        ("beam_plastic", (stations, 6, 2)),
                        ("beam_accumulated", (stations, 2)),
                        ("shell_plastic", (q4_count, 12, 3)),
                        ("shell_alpha", (q4_count, 12)))

            generation, arrays, states = _decode_s3_binary_checkpoint(
                raw, effective_identity, expected_s3_shapes,
                len(self._definition.shell.s3_elements), anchored=chart is not None)
            value = {"schema": FINITE_S3_MATERIAL_CHECKPOINT,
                     "model": effective_identity, "generation": generation,
                     **arrays, "s3_states": states,
                     "accepted_factors": arrays["accepted_factors"].tolist()}
        else:
            if len(raw) > MAX_JSON_CHECKPOINT_BYTES:
                raise ValueError("bounded mixed linear checkpoint required")
            try:
                value = json.loads(raw.decode("ascii"))
            except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
                raise ValueError("invalid mixed linear checkpoint") from exc
            if (type(value) is dict and value.get("schema") in (
                    "B3_GE_E4_PL_STIFFENED_SHELL_EXPERIMENTAL_NATIVE_MATERIAL_V7", NATIVE_SHELL_SCHEMA, STABLE_NATIVE_SCHEMA,
                    _BINARY_SCHEMAS["finite_compensated_generalized_material"],
                    _BINARY_SCHEMAS["finite_compensated_fibre_material"],
                    _BINARY_SCHEMAS["finite_chart_compensated_fibre_material"],
                    _BINARY_SCHEMAS["finite_compensated_mixed_material"], _BINARY_SCHEMAS["finite_compensated_generalized_coupled"], _BINARY_SCHEMAS["finite_compensated_fibre_coupled"], _BINARY_SCHEMAS["finite_compensated_mixed_coupled"],
                    _BINARY_SCHEMAS["finite_chart_compensated_mixed_material"],
                    _BINARY_SCHEMAS["finite_chart_compensated_generalized_material"],
                    _BINARY_SCHEMAS["finite_chart_compensated_generalized_coupled"],
                    _BINARY_SCHEMAS["finite_chart_compensated_fibre_coupled"],
                    _BINARY_SCHEMAS["finite_chart_compensated_mixed_coupled"])):
                raise ValueError("native material checkpoint requires binary framing")
            if (type(value) is dict and value.get("schema") in tuple(
                    _BINARY_SCHEMAS[k] for name in CURVED_POINT_MATERIAL_KINDS for k in (name,name.replace("finite","finite_chart",1)))):
                raise ValueError("curved material predecessor checkpoint requires binary framing")
            if (type(value) is dict and value.get("schema") in (
                    _BINARY_SCHEMAS["finite_bl5"], _BINARY_SCHEMAS["finite_chart_bl5"],
                    _BINARY_SCHEMAS["finite_bl5_material"], _BINARY_SCHEMAS["finite_chart_bl5_material"],
                    _BINARY_SCHEMAS["finite_bl5_coupled"], _BINARY_SCHEMAS["finite_chart_bl5_coupled"])):
                raise ValueError("finite BL5 checkpoint requires binary framing")
        if (type(value) is dict and value.get("schema") in
                (_BINARY_SCHEMAS["finite_compensated_generalized_material"], _BINARY_SCHEMAS["finite_compensated_fibre_material"], _BINARY_SCHEMAS["finite_compensated_mixed_material"], _BINARY_SCHEMAS["finite_compensated_generalized_coupled"], _BINARY_SCHEMAS["finite_compensated_fibre_coupled"], _BINARY_SCHEMAS["finite_compensated_mixed_coupled"], *tuple(_BINARY_SCHEMAS[k] for k in CURVED_POINT_MATERIAL_KINDS + FINITE_DISTRIBUTING_MATERIAL_KINDS), _BINARY_SCHEMAS["finite_distributing"], FINITE_CHECKPOINT, FINITE_MATERIAL_CHECKPOINT,
                 _BINARY_SCHEMAS["finite_bl5"], _BINARY_SCHEMAS["finite_bl5_material"], _BINARY_SCHEMAS["finite_bl5_coupled"],
                 FINITE_SHELL_MATERIAL_CHECKPOINT,
                 FINITE_S3_MATERIAL_CHECKPOINT,
                  NATIVE_SHELL_SCHEMA,
                  STABLE_NATIVE_SCHEMA,
                 "B3_GE_E4_PL_STIFFENED_SHELL_EXPERIMENTAL_NATIVE_MATERIAL_V7")):
            if self._has_physical_fibre and value["schema"] not in (
                    _BINARY_SCHEMAS["finite_compensated_fibre_material"], _BINARY_SCHEMAS["finite_compensated_mixed_material"], _BINARY_SCHEMAS["finite_compensated_generalized_coupled"], _BINARY_SCHEMAS["finite_compensated_fibre_coupled"], _BINARY_SCHEMAS["finite_compensated_mixed_coupled"], NATIVE_SHELL_SCHEMA, STABLE_NATIVE_SCHEMA, "B3_GE_E4_PL_STIFFENED_SHELL_EXPERIMENTAL_NATIVE_MATERIAL_V7") and not (
                    self._distributing_native and
                    value["schema"] == _BINARY_SCHEMAS[self._distributing_checkpoint_kind]):
                raise ValueError("physical-fibre mixed owner admits reference-linear checkpoints only")
            if (bool(self._finite_bl5 and not self._bl5_native_fibre) != (value["schema"] in (
                    _BINARY_SCHEMAS["finite_bl5"], _BINARY_SCHEMAS["finite_bl5_material"], _BINARY_SCHEMAS["finite_bl5_coupled"])) or
                    self._bl5_material != (value["schema"] in (_BINARY_SCHEMAS["finite_bl5_material"], _BINARY_SCHEMAS["finite_bl5_coupled"])) or
                    self._bl5_shell_material != (value["schema"] == _BINARY_SCHEMAS["finite_bl5_coupled"])):
                raise ValueError("finite checkpoint requires matching BL5 mechanics identity")
            if self._line_samples and not self._finite_bl5:
                raise ValueError(
                    "experimental weak line admits reference-linear checkpoints only")
            from ._stiffened_shell_ge_finite_experimental import FinitePointDriver, supports_paired_fibre_adaptive
            if adaptive is not None:
                allow_fibre_predictor = bool(selected and corrections is not None and
                    getattr(self, '_staged_context', None) is not None and
                    value['schema'] == _BINARY_SCHEMAS['finite_compensated_fibre_material'] and
                    supports_paired_fibre_adaptive(self))
                # Narrow coupled-profile accounting admission: only the
                # coupled BL5 J2 mechanics schema on its matching owner may
                # replay predictor work counters; every other profile keeps
                # the historical/default or fibre rules unchanged.
                allow_coupled_predictor = bool(
                    value['schema'] in (_BINARY_SCHEMAS['finite_bl5_coupled'],
                                        _BINARY_SCHEMAS['finite_chart_bl5_coupled']) and
                    self._bl5_material and self._bl5_shell_material)
                validate_control(adaptive, value["accepted_factors"],
                                 _allow_fibre_predictor=allow_fibre_predictor,
                                 _allow_coupled_predictor=allow_coupled_predictor)
                if self._finite_distributing:
                    # Generic replay also supports decreasing sequences; this
                    # profile must admit the complete sequence before publish.
                    origin = adaptive.attempts[0].accepted_from
                    if any(b <= a for a, b in zip(
                            (origin,) + adaptive.targets, adaptive.targets)):
                        raise ValueError(
                            "finite distributing adaptive checkpoint requires increasing force targets")
            driver = FinitePointDriver(self)
            result = (driver.restore(value, adaptive_state=adaptive, stable_q4=stable_q4, corrections=corrections) if chart is None else
                      driver.restore(value, rotation_anchors=chart[1],
                                     pose_witness=chart[2], accepted_generation=chart[0],
                                     adaptive_state=adaptive, stable_q4=stable_q4, corrections=corrections))
            self._finite_driver = driver
            return result
        if adaptive is not None:
            raise ValueError("adaptive checkpoint requires a finite point state")
        if (type(value) is not dict or set(value) !=
                {"schema", "model", "generation", "q", "multipliers", "load"} or
                value["schema"] != CHECKPOINT or value["model"] != effective_identity or
                type(value["generation"]) is not int or value["generation"] < 1):
            # The selected polishing candidate admits only its tagged bytes,
            # and its tagged bytes restore only with the policy selected;
            # both cross-policy directions refuse explicitly.
            if (polish_selected and type(value) is dict and
                    value.get("model") == self._identity):
                raise ValueError(
                    "admissible polish candidate requires its tagged checkpoint")
            if (not polish_selected and type(value) is dict and
                    value.get("model") == admissible_polish_identity(self._identity)):
                raise ValueError(
                    "admissible polish checkpoint requires its selected candidate")
            raise ValueError("incompatible mixed linear checkpoint")
        q = _owned(value["q"], (self._K.shape[0],))
        lam = _owned(value["multipliers"], (self._C.shape[0],))
        load = _owned(value["load"], (self._K.shape[0],))
        if self._bl5_native_material:
            nodal = self._shell_count+6*len(self._beam_ids)
            if np.any(load[nodal:]) or np.any(load[:nodal].reshape(-1, 6)[:, 3:]):
                raise ValueError("native material BL5 force-controlled checkpoints require nodal dead translations")
        if self._distributing_reference and np.any(load[self._shell_count+6*len(self._beam_ids):]):
            raise ValueError("distributing reference checkpoint requires nodal dead actions; retained-coordinate loads are unsupported")
        self._require_curved_load(load)
        equilibrium = self._K@q+self._C.T@lam-load
        if getattr(getattr(self, '_external_context', None), 'pressure_vector_equilibrium', False):
            from ._stiffened_shell_ge_finite_experimental import _pressure_vector_equilibrium
            relative = _pressure_vector_equilibrium(
                equilibrium, load, self._free, self._shell_count,
                self._coordinate_scale)
        else:
            relative = _linear_equilibrium(
                equilibrium[self._free], load[self._free],
                self._coordinate_scale[self._free])
        constraint = _scaled_constraint_gap(
            self._C@q, self._constraint_scale)
        if (not np.isfinite(relative) or relative > 1e-8 or constraint > 1e-9 or
                np.max(np.abs(q[self._fixed])) > 1e-12):
            raise ValueError("mechanically inconsistent mixed linear checkpoint")
        support = np.zeros_like(equilibrium)
        support[self._fixed] = equilibrium[self._fixed]
        result = LinearResult("completed", _owned(q[:self._shell_count]),
                              _owned(q[self._shell_count:]), lam,
                              _owned(support), relative, constraint,
                              value["generation"],
                              point_wrenches=reference_point_records(self, q, lam),
                              distributing_footprint_wrenches=distributing_reference_records(self, q, lam, value["generation"]))
        self._accepted = (result, load)
        self._generation = value["generation"]
        return result

    @_state_writer
    @_accepted_snapshot
    def close(self):
        self._accepted = None
        self._finite_driver = None
        self._linear_factor = None
        self._external_context = None
        self._displacement_snapshot = None
        self._displacement_adaptive = None
        self._arc_snapshot = None
        self._closed = True
