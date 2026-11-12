"""Selected-node six-pose footprints for the elastic reference-linear B2 owner.

This law uses physical nodal translation/rotation coordinates, not the historical
finite-vertex translation fit. It locks normal spin. Complete-cell membership is
an explicit admission policy, not an arbitrary area weld or rigidity theorem.
"""
from dataclasses import dataclass
from math import fsum, isfinite

import numpy as np

LAW_ID = "B2_NODAL_RIGID_FOOTPRINT_REFERENCE_LINEAR_V1_EXPERIMENTAL"


def _identifier(value):
    if type(value) is not str or not value or value != value.strip():
        raise ValueError("nonempty explicit B2 footprint identifier required")
    return value


def _nodes(values, minimum):
    try:
        made = tuple(values)
    except TypeError as exc:
        raise ValueError("owned distinct positive B2 footprint node IDs required") from exc
    if (len(made) < minimum or any(type(v) is not int or v <= 0 for v in made)
            or len(set(made)) != len(made)):
        raise ValueError("owned distinct positive B2 footprint node IDs required")
    return made


@dataclass(frozen=True)
class B2NodalRigidFootprint:
    """An ordered node group at one existing physical B2 endpoint node.

    Omitted shell nodes remain free. A declared patch must contain complete,
    edge-connected Q4 cells and select exactly their vertex union.
    """
    footprint_id: str
    shell_nodes: tuple[int, ...]
    beam_node: int
    patch_cells: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self):
        _identifier(self.footprint_id)
        object.__setattr__(self, "shell_nodes", _nodes(self.shell_nodes, 3))
        if type(self.beam_node) is not int or self.beam_node <= 0:
            raise ValueError("one existing physical B2 endpoint node required")
        try:
            cells = tuple(_nodes(cell, 4) for cell in self.patch_cells)
        except TypeError as exc:
            raise ValueError("explicit B2 footprint Q4 patch cells required") from exc
        if (any(len(cell) != 4 for cell in cells) or
                len({frozenset(cell) for cell in cells}) != len(cells)):
            raise ValueError("distinct explicit B2 footprint Q4 patch cells required")
        object.__setattr__(self, "patch_cells", cells)

    def descriptor(self):
        return dict(law=LAW_ID, footprint_id=self.footprint_id,
                    shell_nodes=self.shell_nodes, beam_node=self.beam_node,
                    patch_cells=self.patch_cells,
                    semantics=("complete_selected_cell_patch" if self.patch_cells
                               else "selected_nodes_only"),
                    configuration="reference_linear", normal_spin="locked",
                    production_qualified=False)


@dataclass(frozen=True)
class B2NodalRigidFootprintSelection:
    footprint_ids: tuple[str, ...]

    def __post_init__(self):
        if isinstance(self.footprint_ids, (str, bytes)):
            raise ValueError("explicit B2 footprint recovery ID sequence required")
        try:
            made = tuple(_identifier(v) for v in self.footprint_ids)
        except TypeError as exc:
            raise ValueError("explicit B2 footprint recovery IDs required") from exc
        if not made or len(set(made)) != len(made):
            raise ValueError("distinct explicit B2 footprint recovery IDs required")
        object.__setattr__(self, "footprint_ids", made)


@dataclass(frozen=True)
class B2NodalFootprintNodeAction:
    shell_node_id: int
    beam_node_id: int
    row_start: int
    shell_reference_position_m: tuple[float, ...]
    beam_reference_position_m: tuple[float, ...]
    shell_force_N: tuple[float, ...]
    shell_direct_couple_Nm: tuple[float, ...]
    beam_force_N: tuple[float, ...]
    beam_direct_couple_Nm: tuple[float, ...]
    gap: tuple[float, ...]


@dataclass(frozen=True)
class B2NodalRigidFootprintWrench:
    """Exerted minus-transpose actions, not multipliers or ground reactions.

    All moments refer to the same reference beam-node origin. Per-node direct
    couples remain visible, including normal-spin/drilling dependence.
    """
    footprint_id: str
    common_reference_origin_m: tuple[float, ...]
    shell_force_N: tuple[float, ...]
    shell_moment_Nm: tuple[float, ...]
    beam_force_N: tuple[float, ...]
    beam_moment_Nm: tuple[float, ...]
    node_actions: tuple[B2NodalFootprintNodeAction, ...]
    law_id: str = LAW_ID
    production_qualified: bool = False


@dataclass(frozen=True)
class _Incidence:
    footprint_id: str
    shell_node: int
    beam_node: int
    row_start: int
    shell_slots: tuple[int, ...]
    beam_slots: tuple[int, ...]
    shell_position: tuple[float, ...]
    beam_position: tuple[float, ...]
    shell_matrix: tuple[tuple[float, ...], ...]


def _difference(a, b):
    high = a-b
    if not isfinite(high):
        raise ValueError("B2 footprint reference arm is not representable")
    return high, fsum((a, -b, -high))


def reference_shell_matrix(shell_position, beam_position):
    """g=(ub-us+(Xb-Xs) cross theta_s, theta_b-theta_s)."""
    d = np.array([fsum((float(b), -float(s)))
                  for s, b in zip(shell_position, beam_position)])
    if d.shape != (3,) or not np.isfinite(d).all():
        raise ValueError("finite representable B2 footprint reference arm required")
    x, y, z = d
    cross = np.array(((0., -z, y), (z, 0., -x), (-y, x, 0.)))
    result = -np.eye(6)
    result[:3, 3:] = cross
    return result


def compile_footprints(footprints, positions, shells, beam_positions,
                       shell_index, beam_index, points):
    """Authenticate membership and compile ordered immutable nodal incidences."""
    try:
        supplied = tuple(footprints)
    except TypeError as exc:
        raise ValueError("typed experimental B2 nodal footprints required") from exc
    if any(type(g) is not B2NodalRigidFootprint for g in supplied):
        raise ValueError("typed experimental B2 nodal footprints required")
    groups = tuple(B2NodalRigidFootprint(g.footprint_id, g.shell_nodes,
                                       g.beam_node, g.patch_cells) for g in supplied)
    return _compile_audited_footprints(groups, positions, shells, beam_positions,
                                       shell_index, beam_index, points, cell_label="Q4")


def _compile_audited_footprints(groups, positions, shells, beam_positions,
                                shell_index, beam_index, points, *, cell_label):
    """Shared membership audit; callers own record and cell-family admission.

    The returned six-slot incidences are geometric records, not adoption of a
    constraint law. Physical footprints replace both matrices and row offsets.
    """
    known_cells = {frozenset(s.node_ids): s.node_ids for s in shells}
    shell_hosts = {s.shell_id: s.node_ids for s in shells}
    seen_nodes = {n: f"finite-vertex point={p.point_id}"
                  for p in points for n in shell_hosts[p.shell_id]}
    seen_ids = set()
    point_ids = {p.point_id for p in points}
    incidences = []
    for group in groups:
        label = f"footprint={group.footprint_id} beam={group.beam_node}"
        if group.footprint_id in seen_ids or group.footprint_id in point_ids:
            raise ValueError(f"duplicate or point-colliding B2 footprint ID: {label}")
        seen_ids.add(group.footprint_id)
        if group.beam_node not in beam_positions or any(n not in positions for n in group.shell_nodes):
            raise ValueError(f"known shell nodes and physical B2 endpoint required: {label}")
        base = positions[group.shell_nodes[0]]
        differences = np.array([[fsum((float(positions[n][a]), -float(base[a])))
                                 for a in range(3)] for n in group.shell_nodes])
        scale = float(np.max(np.abs(differences)))
        if (not np.isfinite(differences).all() or scale == 0. or
                np.linalg.matrix_rank(differences/scale) < 2):
            raise ValueError(f"noncollinear B2 footprint shell nodes required: {label}")
        if group.patch_cells:
            cells = tuple(frozenset(c) for c in group.patch_cells)
            if any(c not in known_cells for c in cells):
                raise ValueError(f"known complete {cell_label} footprint cells required: {label}")
            if set(group.shell_nodes) != set().union(*cells):
                raise ValueError(f"every cell vertex and only patch vertices required: {label}")
            edges = [{frozenset((c[i], c[(i+1) % len(c)])) for i in range(len(c))}
                     for c in (known_cells[key] for key in cells)]
            reached = {0}; pending = [0]
            while pending:
                current = pending.pop()
                for other in range(len(cells)):
                    if other not in reached and edges[current] & edges[other]:
                        reached.add(other); pending.append(other)
            if len(reached) != len(cells):
                raise ValueError(f"edge-connected complete {cell_label} footprint patch required: {label}")
        for node in group.shell_nodes:
            if node in seen_nodes:
                raise ValueError(f"overlapping B2 footprint shell={node}: {label}; {seen_nodes[node]}")
            seen_nodes[node] = label
            sp, bp = tuple(map(float, positions[node])), tuple(map(float, beam_positions[group.beam_node]))
            matrix = reference_shell_matrix(sp, bp)
            incidences.append(_Incidence(
                group.footprint_id, node, group.beam_node,
                6*(len(points)+len(incidences)),
                tuple(6*shell_index[node]+j for j in range(6)),
                tuple(6*beam_index[group.beam_node]+j for j in range(6)),
                sp, bp, tuple(tuple(map(float, row)) for row in matrix)))
    return groups, tuple(incidences)


def checked_selection(selection, groups):
    if selection is None:
        return ()
    if type(selection) is not B2NodalRigidFootprintSelection:
        raise ValueError("typed B2 nodal footprint recovery selection required")
    available = {g.footprint_id for g in groups}
    if any(name not in available for name in selection.footprint_ids):
        raise ValueError("selected B2 nodal footprint is not compiled")
    return selection.footprint_ids


def selected_wrenches(incidences, selected, multipliers, gaps):
    """Build only requested, detached actual transpose-action records."""
    made = []
    for name in selected:
        actions = []
        forces = [[], [], []]; moments = [[], [], []]
        beam_forces = [[], [], []]; beam_moments = [[], [], []]
        origin = None
        for item in incidences:
            if item.footprint_id != name:
                continue
            origin = item.beam_position
            lam = multipliers[item.row_start:item.row_start+6]
            shell = -np.asarray(item.shell_matrix).T @ lam
            beam = -lam
            arm = [_difference(s, b) for s, b in zip(item.shell_position, origin)]
            for axis in range(3):
                forces[axis].append(float(shell[axis]))
                beam_forces[axis].append(float(beam[axis]))
                beam_moments[axis].append(float(beam[axis+3]))
                j, k = (axis+1) % 3, (axis+2) % 3
                moments[axis].extend((float(shell[axis+3]),
                    arm[j][0]*shell[k], arm[j][1]*shell[k],
                    -arm[k][0]*shell[j], -arm[k][1]*shell[j]))
            actions.append(B2NodalFootprintNodeAction(
                item.shell_node, item.beam_node, item.row_start,
                item.shell_position, item.beam_position,
                tuple(map(float, shell[:3])), tuple(map(float, shell[3:])),
                tuple(map(float, beam[:3])), tuple(map(float, beam[3:])),
                tuple(map(float, gaps[item.row_start:item.row_start+6]))))
        sums = tuple(tuple(fsum(component) for component in vectors)
                     for vectors in (forces, moments, beam_forces, beam_moments))
        if (origin is None or not all(isfinite(x) for vector in sums for x in vector)
                or any(not isfinite(x) for action in actions for vector in (
                    action.shell_force_N, action.shell_direct_couple_Nm,
                    action.beam_force_N, action.beam_direct_couple_Nm, action.gap)
                       for x in vector)):
            raise ValueError("finite B2 footprint transpose-action recovery required")
        made.append(B2NodalRigidFootprintWrench(name, origin, *sums, tuple(actions)))
    return tuple(made)
