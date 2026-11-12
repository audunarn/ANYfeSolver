"""Explicit experimental selected-node rigid footprints, not area welds.

Each selected shell pose uses the existing six-row objective point law at one
native beam node. Declared complete-patch admission requires all cell vertices;
this is a product policy, not a universal rigidity criterion. No distributing
weights, fit, area quadrature or new law occurs. The default unrebased policy
retains its chart guards. Opt-in elastic V2 increments use one detached physical
origin chart, held fixed throughout Newton/backtracking.
"""
from dataclasses import dataclass, field
from math import fsum, isfinite

import numpy as np
from ._stiffened_shell_point_recovery import PointTransferWrench

LAW = "GE_NODAL_RIGID_FOOTPRINT_POINT_POSE_V1_EXPERIMENTAL"
STABLE_S3_POLICY = "GE_NODAL_RIGID_FOOTPRINT_ELASTIC_S3_STABLE_NATIVE_HISTORY_V1"
RETAINED_SCALE_POLICY = "GE_NODAL_RIGID_FOOTPRINT_RETAINED_FORCE6_MOMENT12_WORK_SCALE_V1"
CHART_ADMISSION_POLICY = "GE_NODAL_RIGID_FOOTPRINT_UNREBASED_ADMISSION_V1"
INCREMENT_ORIGIN_CHART_POLICY = "GE_NODAL_RIGID_FOOTPRINT_INCREMENT_ORIGIN_CHART_V2"


def chart_admission_policy_descriptor(policy=CHART_ADMISSION_POLICY):
    if policy == CHART_ADMISSION_POLICY:
        return dict(policy=CHART_ADMISSION_POLICY,
                    rebase="unsupported", chart_checkpoint_restore="unsupported",
                    reason="footprint_chart_continuation_parity_not_established",
                    production_qualified=False)
    if policy != INCREMENT_ORIGIN_CHART_POLICY:
        raise ValueError("explicit supported footprint chart policy required")
    return dict(policy=INCREMENT_ORIGIN_CHART_POLICY,
                rebase="experimental_elastic", chart_checkpoint_restore="experimental_elastic",
                increment_origin="zero_rotation_coordinates_at_accepted_physical_frames",
                newton_anchor="fixed_for_all_iterations_and_backtracks",
                rotational_supports="fully_fixed_or_fully_free_triples",
                production_qualified=False)


def require_rotation_triples(fixed, shell_nodes):
    """A chart reset cannot reinterpret a partially fixed coordinate triple."""
    fixed = set(fixed)
    for node in range(shell_nodes):
        restrained = [6*node+j in fixed for j in (3, 4, 5)]
        if any(restrained) and not all(restrained):
            raise ValueError("nodal footprint finite charts require full rotational triples; partial supports need transformed equations")


def retained_scale_policy_descriptor():
    return dict(policy=RETAINED_SCALE_POLICY,
                retained_resultants="two_force_triplets_then_four_moment_triplets",
                scale_source="absolute_virgin_section_tangent_times_reference_strain_scale",
                legacy_empty_footprints_unchanged=True)


def stable_s3_policy_descriptor():
    return dict(policy=STABLE_S3_POLICY,
                s3_translation="cancellation_resistant_reference_difference",
                beam_history="complete_native_origin_and_accepted_compensated_pairs",
                elastic_response_only=True)


def _identifier(value):
    if type(value) is not str or not value or value != value.strip():
        raise ValueError("nonempty explicit footprint identifier required")
    return value


def _nodes(values, minimum):
    try:
        made = tuple(values)
    except TypeError as exc:
        raise ValueError("owned distinct positive footprint node IDs required") from exc
    if (len(made) < minimum or any(type(v) is not int or v <= 0 for v in made)
            or len(set(made)) != len(made)):
        raise ValueError("owned distinct positive footprint node IDs required")
    return made


@dataclass(frozen=True)
class NodalRigidFootprint:
    """An ordered selected-node rigid body at an existing native beam node.

``patch_cells`` optionally declares a complete connected Q4/S3 cell patch.
Then the selected nodes must be exactly its vertex union. Without this field
only the listed nodes are constrained; omitted element nodes remain free.
The full six-channel pose law locks normal spin and is not BL5 or a weld.
"""
    footprint_id: str
    shell_nodes: tuple[int, ...]
    beam_node: int
    patch_cells: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self):
        _identifier(self.footprint_id)
        object.__setattr__(self, "shell_nodes", _nodes(self.shell_nodes, 3))
        if type(self.beam_node) is not int or self.beam_node <= 0:
            raise ValueError("one existing native beam nodal station required")
        try:
            cells = tuple(_nodes(cell, 3) for cell in self.patch_cells)
        except TypeError as exc:
            raise ValueError("explicit Q4/S3 footprint patch cells required") from exc
        if (any(len(cell) not in (3, 4) for cell in cells) or
                len({frozenset(cell) for cell in cells}) != len(cells)):
            raise ValueError("distinct explicit Q4/S3 footprint patch cells required")
        object.__setattr__(self, "patch_cells", cells)

    def descriptor(self):
        return dict(law=LAW, footprint_id=self.footprint_id,
                    shell_nodes=self.shell_nodes, beam_node=self.beam_node,
                    patch_cells=self.patch_cells,
                    semantics=("complete_selected_cell_patch" if self.patch_cells
                               else "selected_nodes_only"))


@dataclass(frozen=True)
class NodalRigidFootprintSelection:
    """Recover explicitly named groups without changing default recovery."""
    footprint_ids: tuple[str, ...]

    def __post_init__(self):
        if isinstance(self.footprint_ids, (str, bytes)):
            raise ValueError("explicit footprint recovery identifier sequence required")
        try:
            made = tuple(_identifier(v) for v in self.footprint_ids)
        except TypeError as exc:
            raise ValueError("explicit footprint recovery identifiers required") from exc
        if not made or len(set(made)) != len(made):
            raise ValueError("distinct explicit footprint recovery identifiers required")
        object.__setattr__(self, "footprint_ids", made)


@dataclass(frozen=True)
class NodalRigidFootprintWrench:
    """Exerted domain actions, both about the same beam-port origin.

Per-node records preserve each original port, low position, action and gap.
These are sums of actual minus-transpose residuals, not spatial multipliers
or ground reactions. Reference-linear and current origins remain distinct.
"""
    footprint_id: str
    beam_node_id: int
    shell_node_ids: tuple[int, ...]
    configuration: str
    common_origin_m: tuple[float, float, float]
    common_origin_low_m: tuple[float, float, float]
    shell_action: tuple[float, float, float, float, float, float]
    beam_action: tuple[float, float, float, float, float, float]
    point_wrenches: tuple[PointTransferWrench, ...]
    generation: int
    law: str = field(default=LAW, init=False)
    production_qualified: bool = field(default=False, init=False)


def validate_footprints(footprints, shell_index, shell_xyz, shell_cells,
                        beam_index, points, lines):
    """Own and authenticate groups; return ordered, disjoint node incidences."""
    try:
        groups = tuple(footprints)
    except TypeError as exc:
        raise ValueError("typed experimental nodal rigid footprints required") from exc
    if any(type(group) is not NodalRigidFootprint for group in groups):
        raise ValueError("typed experimental nodal rigid footprints required")
    groups = tuple(NodalRigidFootprint(g.footprint_id, g.shell_nodes,
                                      g.beam_node, g.patch_cells) for g in groups)
    if not groups:
        return groups
    seen_ids = set()
    seen_nodes = {p.shell_node: f"point shell={p.shell_node} beam={p.beam_node}"
                  for p in points}
    for line_index, line in enumerate(lines):
        for node in line.shell_nodes:
            seen_nodes[node] = f"weak line={line_index} shell={node}"
    known_cells = {frozenset(cell): cell for cell in shell_cells}
    for group in groups:
        label = f"footprint={group.footprint_id} beam={group.beam_node}"
        if group.footprint_id in seen_ids:
            raise ValueError(f"duplicate nodal rigid footprint identifier: {label}")
        seen_ids.add(group.footprint_id)
        if group.beam_node not in beam_index or any(node not in shell_index for node in group.shell_nodes):
            raise ValueError(f"known shell nodes and native beam nodal station required: {label}")
        xy = np.array([[fsum((float(shell_xyz[shell_index[node], a]),
                              -float(shell_xyz[shell_index[group.shell_nodes[0]], a])))
                        for a in (0, 1)] for node in group.shell_nodes])
        span = float(np.max(np.abs(xy)))
        if not np.isfinite(xy).all() or span == 0. or np.linalg.matrix_rank(xy/span) != 2:
            raise ValueError(f"noncollinear footprint shell nodes required: {label}")
        if group.patch_cells:
            cells = tuple(frozenset(cell) for cell in group.patch_cells)
            if any(cell not in known_cells for cell in cells):
                raise ValueError(f"known complete Q4/S3 footprint patch cells required: {label}")
            if set(group.shell_nodes) != set().union(*cells):
                raise ValueError(f"every cell vertex and only patch vertices must be selected: {label}")
            edges = [{frozenset((cell[i], cell[(i+1) % len(cell)]))
                      for i in range(len(cell))} for cell in
                     (known_cells[key] for key in cells)]
            reached = {0}; pending = [0]
            while pending:
                current = pending.pop()
                for other in range(len(cells)):
                    if other not in reached and edges[current] & edges[other]:
                        reached.add(other); pending.append(other)
            if len(reached) != len(cells):
                raise ValueError(f"connected complete footprint cell patch required: {label}")
        for node in group.shell_nodes:
            if node in seen_nodes:
                raise ValueError(f"overlapping nodal rigid footprint shell={node}: {label}; {seen_nodes[node]}")
            seen_nodes[node] = label
    return groups


def selected_wrenches(owner, result, selection):
    """Sum already published point actions; no mechanics replay or mutation."""
    if type(selection) is not NodalRigidFootprintSelection:
        raise ValueError("typed nodal rigid footprint recovery selection required")
    groups = {group.footprint_id: (group, indices)
              for group, indices in owner._footprint_bindings}
    if any(name not in groups for name in selection.footprint_ids):
        raise ValueError("recovery references unknown nodal rigid footprint")
    recovered = []
    for name in selection.footprint_ids:
        group, indices = groups[name]
        records = tuple(result.point_wrenches[i] for i in indices)
        origin, low = records[0].beam_position_m, records[0].beam_position_low_m
        configuration = records[0].configuration
        if any(r.beam_node_id != group.beam_node or r.beam_position_m != origin
               or r.beam_position_low_m != low or r.configuration != configuration
               for r in records):
            raise ValueError("inconsistent accepted footprint port recovery")
        def total(shell):
            force_terms = [[] for _ in range(3)]; moment_terms = [[] for _ in range(3)]
            for r in records:
                action = r.shell_action if shell else r.beam_action
                position = r.shell_position_m if shell else r.beam_position_m
                position_low = r.shell_position_low_m if shell else r.beam_position_low_m
                # Keep low lever arms separate through multiplication: adding
                # 2**-54 to a1m arm first would erase its finite force moment.
                arm = [fsum((position[a], -origin[a])) for a in range(3)]
                arm_low = [fsum((position_low[a], -low[a])) for a in range(3)]
                for a in range(3):
                    b, c = (a+1) % 3, (a+2) % 3
                    force_terms[a].append(action[a])
                    moment_terms[a].extend((action[a+3], arm[b]*action[c],
                                            -arm[c]*action[b], arm_low[b]*action[c],
                                            -arm_low[c]*action[b]))
            try:
                values = tuple(fsum(v) for v in force_terms+moment_terms)
            except (OverflowError, ValueError) as exc:
                raise ValueError("finite physical footprint resultants required") from exc
            if not all(isfinite(v) for v in values):
                raise ValueError("finite physical footprint resultants required")
            return values
        recovered.append(NodalRigidFootprintWrench(
            name, group.beam_node, group.shell_nodes, configuration,
            origin, low, total(True), total(False), records, result.generation))
    return tuple(recovered)
