"""Sparse reference-linear assembly of unqualified L1 and BL5 weak lines.

The definition explicitly pairs material shell and beam traces. BL5 splits
straight reference lines at both meshes' boundaries; L1 remains matching-only.
This is not reference search, a selected multiplier policy, or a qualified bond.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from ._ge_beam3_retained_generalized_state import Mechanical
from ._stiffened_shell_l1_local_experimental import (
    L1LocalOperator, NativeHalfCellPort,
)
from ._stiffened_shell_bl5_local_experimental import BL5LocalOperator, LAW_ID


SPACE_A = "P1_translation_twist_P0_pitch_normal"
SPACE_B = "P0_translation_twist_pair_P0_pitch_normal"
SPACES = (SPACE_A, SPACE_B)


def compile_supported_shell_work(*, shell_ids, shell_xyz, shell_cells,
                                 trace_nodes, full_count, fixed_slots,
                                 node_slots=None):
    """Compile a declared shell-only work family using actual trace topology.

    This is a development comparison map, not a change to line assembly or its
    physical law. Explicit full slots admit different physical node orderings;
    omitted beam/retained roles remain present and identically zero.
    """
    from ._stiffened_shell_l1_supported_work_experimental import flat_q4_supported_shell_map

    index = {node: i for i, node in enumerate(shell_ids)}
    if (len(index) != len(shell_ids) or len(shell_xyz) != len(shell_ids) or
            any(node not in index for cell in shell_cells for node in cell) or
            any(node not in index for node in trace_nodes)):
        raise ValueError("supported work requires the actual shell and trace node identities")
    if node_slots is None:
        node_slots = np.arange(6*len(shell_ids)).reshape((-1, 6))
    return flat_q4_supported_shell_map(
        coordinates=shell_xyz, node_slots=node_slots, full_count=full_count,
        fixed_slots=fixed_slots, trace_nodes=tuple(index[node] for node in trace_nodes),
        cells=tuple(tuple(index[node] for node in cell) for cell in shell_cells))


@dataclass(frozen=True)
class ExperimentalL1Line:
    """One explicitly mapped material line for reference-linear experiments."""

    shell_nodes: tuple[int, ...]
    beam_halfcells: tuple[tuple[int, int], ...]
    multiplier_space: str
    beam_offset: tuple[float, float, float] = (0., 0., -.05)
    expected_shell_offset: tuple[float, float, float] = (0., 0., 0.)

    def __post_init__(self):
        n = len(self.beam_halfcells)
        if (type(self.shell_nodes) is not tuple or
                type(self.beam_halfcells) is not tuple or
                n < 2 or n % 2 or len(self.shell_nodes) != n+1 or
                len(set(self.shell_nodes)) != n+1 or
                any(type(v) is not int or v <= 0 for v in self.shell_nodes) or
                any(type(pair) is not tuple or len(pair) != 2 or
                    type(pair[0]) is not int or type(pair[1]) is not int or
                    pair[0] < 0 or pair[1] not in (0, 1)
                    for pair in self.beam_halfcells) or
                len(set(self.beam_halfcells)) != n or
                self.multiplier_space not in SPACES):
            raise ValueError("ordered even-edge experimental L1 line required")
        for values in (self.beam_offset, self.expected_shell_offset):
            if (type(values) is not tuple or len(values) != 3 or
                    not np.isfinite(values).all()):
                raise ValueError("finite declared L1 offset required")

    def descriptor(self):
        return dict(shell_nodes=self.shell_nodes,
                    beam_halfcells=self.beam_halfcells,
                    multiplier_space=self.multiplier_space,
                    beam_offset=self.beam_offset,
                    expected_shell_offset=self.expected_shell_offset)


@dataclass(frozen=True)
class ExperimentalBL5Line:
    """Separate five-row director bond for experimental mixed analyses.

    ``multiplier_span`` counts whole consecutive native half-cells per
    support. It must divide the trace; no final partial support is inferred.
    ``discontinuous_p1`` adds one affine density mode on each physical support,
    shared by all five channels. Those supports must contain complete members.
    ``discontinuous_p1_cell_average`` uses that affine mode's reference
    arclength average on each native half-cell, for all five channels. Its
    density is constant within a half-cell, including split shell segments.
    Choosing that count does not qualify an interface-resolution policy.
    """

    shell_nodes: tuple[int, ...]
    beam_halfcells: tuple[tuple[int, int], ...]
    multiplier_span: int = 1
    expected_beam_arm: tuple[float, float, float] = (0., 0., -.05)
    basis: str = "p0"

    def __post_init__(self):
        n = len(self.beam_halfcells)
        if (type(self.shell_nodes) is not tuple or
                type(self.beam_halfcells) is not tuple or
                n < 2 or n % 2 or len(self.shell_nodes) < 3 or
                len(set(self.shell_nodes)) != len(self.shell_nodes) or
                any(type(v) is not int or v <= 0 for v in self.shell_nodes) or
                any(type(pair) is not tuple or len(pair) != 2 or
                    type(pair[0]) is not int or type(pair[1]) is not int or
                    pair[0] < 0 or pair[1] not in (0, 1)
                    for pair in self.beam_halfcells) or
                len(set(self.beam_halfcells)) != n or
                type(self.multiplier_span) is not int or
                not 1 <= self.multiplier_span <= n or
                n % self.multiplier_span or
                type(self.basis) is not str or
                self.basis not in ("p0", "discontinuous_p1", "continuous_p1",
                                   "continuous_p1_coarse",
                                   "discontinuous_p1_cell_average") or
                (self.basis in ("continuous_p1", "continuous_p1_coarse") and
                 self.multiplier_span != 1) or
                type(self.expected_beam_arm) is not tuple or
                len(self.expected_beam_arm) != 3 or
                any(isinstance(v, (bool, np.bool_)) or
                    not isinstance(v, (int, float, np.integer, np.floating)) or
                    not np.isfinite(v) for v in self.expected_beam_arm)):
            raise ValueError("ordered even-edge experimental BL5 line required")
        if self.basis in ("discontinuous_p1", "discontinuous_p1_cell_average") and (
                self.multiplier_span < 2 or self.multiplier_span % 2 or
                any(left[1] != 0 or right != (left[0], 1)
                    for left, right in zip(self.beam_halfcells[::2],
                                           self.beam_halfcells[1::2]))):
            raise ValueError(
                "BL5 discontinuous-P1 supports require complete native members")
        if (self.basis == "continuous_p1" and
                n > len(self.shell_nodes)-1):
            # This beam-node basis has dependent pitch rows on tested
            # beam-finer traces. The large sparse structural audit cannot
            # certify their numerical independence, so fail closed pending
            # a separately derived off-grid resolution rule.
            raise ValueError(
                "experimental beam-node continuous-P1 requires shell "
                "trace resolution at least as fine as beam half-cells")

    @property
    def multiplier_space(self):
        if self.basis == "discontinuous_p1_cell_average":
            return f"BL5_P1_discontinuous_cell_average_span_{self.multiplier_span}"
        if self.basis == "discontinuous_p1":
            return f"BL5_P1_discontinuous_span_{self.multiplier_span}"
        if self.basis == "continuous_p1_coarse":
            return "BL5_P1_coarse_trace"
        if self.basis == "continuous_p1":
            return "BL5_P1_continuous"
        return f"BL5_P0_span_{self.multiplier_span}"

    def descriptor(self):
        result = dict(law=LAW_ID, shell_nodes=self.shell_nodes,
                      beam_halfcells=self.beam_halfcells,
                      multiplier_span=self.multiplier_span,
                      expected_beam_arm=self.expected_beam_arm)
        if self.basis != "p0":
            result["basis"] = self.basis
        return result


def _sample_rows(line: ExperimentalL1Line | ExperimentalBL5Line,
                 shell_edge: int, shell_alpha: float,
                 beam_edge: int, beam_alpha: float,
                 channel: int, support_alpha: float | None = None):
    if type(line) is ExperimentalBL5Line:
        if line.basis in ("discontinuous_p1", "discontinuous_p1_cell_average"):
            if (support_alpha is None or not np.isfinite(support_alpha) or
                    not 0. <= support_alpha <= 1.):
                raise ValueError("physical BL5 support coordinate required")
            supports = len(line.beam_halfcells)//line.multiplier_span
            base = 2*(channel*supports+beam_edge//line.multiplier_span)
            return ((base, 1.), (base+1, 2.*support_alpha-1.))
        if line.basis == "continuous_p1_coarse":
            shell_edges = len(line.shell_nodes)-1
            beam_edges = len(line.beam_halfcells)
            if shell_edges < beam_edges:
                count, edge, alpha = shell_edges, shell_edge, shell_alpha
            else:
                count, edge, alpha = beam_edges, beam_edge, beam_alpha
            base = channel*(count+1)
            return ((base+edge, 1.-alpha), (base+edge+1, alpha))
        if line.basis == "continuous_p1":
            base = channel*(len(line.beam_halfcells)+1)
            return ((base+beam_edge, 1.-beam_alpha),
                    (base+beam_edge+1, beam_alpha))
        return ((channel*(len(line.beam_halfcells)//line.multiplier_span)+
                 beam_edge//line.multiplier_span, 1.),)
    n = len(line.beam_halfcells)
    if line.multiplier_space == SPACE_A:
        if channel < 4:
            base = channel*(n+1)
            return ((base+beam_edge, 1.-beam_alpha),
                    (base+beam_edge+1, beam_alpha))
        return ((4*(n+1)+(channel-4)*n+beam_edge, 1.),)
    if channel < 4:
        return ((channel*n+beam_edge, 1.),)
    return ((4*n+(channel-4)*(n//2)+beam_edge//2, 1.),)


def _row_count(line: ExperimentalL1Line):
    n = len(line.beam_halfcells)
    if type(line) is ExperimentalBL5Line:
        if line.basis in ("discontinuous_p1", "discontinuous_p1_cell_average"):
            return 10*n//line.multiplier_span
        if line.basis == "continuous_p1_coarse":
            return 5*(min(n, len(line.shell_nodes)-1)+1)
        if line.basis == "continuous_p1":
            return 5*(n+1)
        return 5*n//line.multiplier_span
    return 4*(n+1)+2*n if line.multiplier_space == SPACE_A else 4*n+n


def _bl5_segments(line, shell_index, shell_xyz, beam_nodes, beam_parts,
                  quadrature):
    """Intersect two explicitly ordered straight material traces once."""
    shell_points = shell_xyz[[shell_index[node] for node in line.shell_nodes]]
    axis = shell_points[-1]-shell_points[0]
    length = float(np.linalg.norm(axis))
    if not np.isfinite(length) or length <= 1e-10:
        raise ValueError("nondegenerate BL5 material line required")
    axis /= length
    tolerance = 1e-10*length+1e-12
    shell_stations = (shell_points-shell_points[0])@axis
    shell_offaxis = shell_points-shell_points[0]-shell_stations[:, None]*axis
    if (np.max(np.linalg.norm(shell_offaxis, axis=1)) > tolerance or
            np.min(np.diff(shell_stations)) <= tolerance):
        raise ValueError("ordered straight BL5 shell trace required")

    beam_ids = []
    for element_index, cell in line.beam_halfcells:
        if element_index >= len(beam_parts):
            raise ValueError("BL5 line references unknown native beam half-cell")
        pair = beam_parts[element_index][0].node_ids[cell:cell+2]
        if beam_ids and beam_ids[-1] != pair[0]:
            raise ValueError("contiguous ordered BL5 beam half-cells required")
        if not beam_ids:
            beam_ids.append(pair[0])
        beam_ids.append(pair[1])
    beam_points = np.array([beam_nodes[node][0] for node in beam_ids])
    beam_stations = (beam_points-beam_points[0])@axis
    beam_offaxis = beam_points-beam_points[0]-beam_stations[:, None]*axis
    if (np.max(np.linalg.norm(beam_offaxis, axis=1)) > tolerance or
            np.min(np.diff(beam_stations)) <= tolerance or
            abs(beam_stations[-1]-length) > tolerance or
            abs((beam_points[0]-shell_points[0])@axis) > tolerance):
        raise ValueError("aligned straight BL5 beam trace required")

    endpoints = np.sort(np.r_[shell_stations, beam_stations])
    partitions = [0.]
    for value in endpoints[1:]:
        if value-partitions[-1] > tolerance:
            partitions.append(float(value))
    if abs(partitions[-1]-length) > tolerance:
        raise ValueError("complete BL5 reference line coverage required")
    partitions[-1] = length
    for left, right in zip(partitions[:-1], partitions[1:]):
        middle = .5*(left+right)
        shell_edge = min(int(np.searchsorted(shell_stations, middle))-1,
                         len(shell_stations)-2)
        beam_edge = min(int(np.searchsorted(beam_stations, middle))-1,
                        len(beam_stations)-2)
        for gauss in quadrature:
            station = left+gauss*(right-left)
            shell_alpha = ((station-shell_stations[shell_edge]) /
                           (shell_stations[shell_edge+1]-shell_stations[shell_edge]))
            beam_alpha = ((station-beam_stations[beam_edge]) /
                          (beam_stations[beam_edge+1]-beam_stations[beam_edge]))
            support_alpha = None
            if line.basis in ("discontinuous_p1", "discontinuous_p1_cell_average"):
                start = (beam_edge//line.multiplier_span)*line.multiplier_span
                end = start+line.multiplier_span
                density_station = station
                if line.basis == "discontinuous_p1_cell_average":
                    # Average of an affine reference-arclength coordinate:
                    # use native cell endpoints, never the cell index or
                    # this shell/beam integration segment's endpoints.
                    density_station = .5*(beam_stations[beam_edge]+
                                          beam_stations[beam_edge+1])
                support_alpha = ((density_station-beam_stations[start]) /
                                 (beam_stations[end]-beam_stations[start]))
            yield (shell_edge, beam_edge, shell_alpha, beam_alpha,
                   .5*(right-left), support_alpha)


def compile_lines(lines, *, shell_ids, shell_xyz, shell_cells,
                  thickness, beam_ids, beam_nodes, beam_parts,
                  shell_count, beam_count, with_terms=False):
    """Build the fixed sparse C rows and keep local operators for later work.

    The returned rows are derivatives of integrated constraint work with
    respect to multiplier coefficients, not pointwise collocation equations.
    """
    if type(lines) is not tuple or any(type(line) not in
                                       (ExperimentalL1Line, ExperimentalBL5Line)
                                       for line in lines):
        raise ValueError("typed experimental weak lines required")
    if type(with_terms) is not bool:
        raise ValueError("explicit finite line term selection required")
    terms = []
    shell_index = {node: i for i, node in enumerate(shell_ids)}
    beam_index = {node: i for i, node in enumerate(beam_ids)}
    shell_edges = {frozenset((cell[j], cell[(j+1) % len(cell)]))
                   for cell in shell_cells for j in range(len(cell))}
    positions = np.array([beam_nodes[node][0] for node in beam_ids])
    frames = np.array([beam_nodes[node][1] for node in beam_ids])
    state = Mechanical(positions, np.zeros_like(positions), frames,
                       np.tile(np.eye(3), (len(beam_parts), 2, 1, 1)),
                       np.zeros((len(beam_parts), 18)))
    count = shell_count+beam_count
    row_indices, columns, values = [], [], []
    samples = []
    line_row = 0
    seen_lines = set()
    abscissae = (.5-.5/np.sqrt(3.), .5+.5/np.sqrt(3.))
    for line in lines:
        line.__post_init__()
        label = "L1" if type(line) is ExperimentalL1Line else "BL5"
        identity = (line.shell_nodes, line.beam_halfcells)
        if identity in seen_lines:
            raise ValueError(f"repeated experimental {label} material line")
        seen_lines.add(identity)
        if any(node not in shell_index for node in line.shell_nodes):
            raise ValueError(f"{label} line references unknown shell node")
        if any(frozenset((line.shell_nodes[i], line.shell_nodes[i+1]))
               not in shell_edges for i in range(len(line.shell_nodes)-1)):
            raise ValueError(f"{label} material trace must follow shell element edges")
        if type(line) is ExperimentalBL5Line:
            segments = _bl5_segments(
                line, shell_index, shell_xyz, beam_nodes, beam_parts,
                abscissae)
        else:
            segments = ((edge, edge, float(alpha), float(alpha),
                         .5*float(np.linalg.norm(shell_xyz[shell_index[line.shell_nodes[edge+1]]]-
                                                  shell_xyz[shell_index[line.shell_nodes[edge]]])), None)
                        for edge in range(len(line.beam_halfcells))
                        for alpha in abscissae)
        seen_stations = set()
        for shell_edge, beam_edge, shell_alpha, beam_alpha, weight, support_alpha in segments:
            element_index, cell = line.beam_halfcells[beam_edge]
            if element_index >= len(beam_parts):
                raise ValueError(f"{label} line references unknown native beam half-cell")
            element = beam_parts[element_index][0]
            reference = element.operator.reference
            node_rows = tuple(beam_index[node] for node in element.node_ids)
            shell_pair = line.shell_nodes[shell_edge:shell_edge+2]
            shell_rows = tuple(shell_index[node] for node in shell_pair)
            shell_reference = shell_xyz[list(shell_rows)]
            shell_slots = np.r_[np.arange(6*shell_rows[0], 6*shell_rows[0]+6),
                                np.arange(6*shell_rows[1], 6*shell_rows[1]+6)]
            length = float(np.linalg.norm(shell_reference[1]-shell_reference[0]))
            if not np.isfinite(length) or length <= 1e-10:
                raise ValueError(f"nondegenerate {label} reference edge required")
            station = (element_index, cell, float(beam_alpha))
            if station in seen_stations:
                raise ValueError(f"repeated native {label} station")
            seen_stations.add(station)
            port = NativeHalfCellPort.from_reference(
                reference, node_rows, element_index, cell, float(beam_alpha),
                beam_count-24*len(beam_parts), beam_count)
            if type(line) is ExperimentalL1Line:
                local_operator = L1LocalOperator.at_reference(
                    shell_reference, np.zeros((2, 6)), thickness,
                    port, state, offset=line.beam_offset,
                    station=float(shell_alpha))
                observed_offset = local_operator.reference_offset
                expected_offset = line.expected_shell_offset
            else:
                local_operator = BL5LocalOperator.at_reference(
                    shell_reference, np.zeros((2, 6)), thickness,
                    port, state, station=float(shell_alpha))
                observed_offset = local_operator.reference_arm
                expected_offset = line.expected_beam_arm
            if not np.allclose(observed_offset, expected_offset,
                               rtol=1e-10, atol=1e-10):
                raise ValueError(f"{label} reference correspondence offset mismatch")
            local = local_operator.evaluate(
                np.zeros((2, 6)), state,
                np.zeros(6 if type(line) is ExperimentalL1Line else 5))
            if np.max(np.abs(local.gap)) > 2e-13:
                raise ValueError(f"{label} material reference gap is not zero")
            global_slots = np.r_[shell_slots,
                                 shell_count+local.beam_slots]
            row_terms = []
            for channel in range(len(local.gap)):
                for row, coefficient in _sample_rows(
                        line, shell_edge, float(shell_alpha),
                        beam_edge, float(beam_alpha), channel, support_alpha):
                    jacobian = weight*coefficient*local.gap_jacobian[channel]
                    row_indices.extend([line_row+row]*len(global_slots))
                    columns.extend(global_slots.tolist())
                    values.extend(jacobian.tolist())
                    row_terms.append((channel, line_row+row, coefficient))
            samples.append((local_operator, global_slots, weight,
                            beam_edge, float(beam_alpha)))
            if with_terms:
                row_ids = np.array(sorted({row for _, row, _ in row_terms}), dtype=int)
                interpolation = np.zeros((len(local.gap), len(row_ids)))
                for channel, row, coefficient in row_terms:
                    interpolation[channel, np.searchsorted(row_ids, row)] += coefficient
                row_ids.setflags(write=False)
                interpolation.setflags(write=False)
                terms.append((row_ids, interpolation))
        line_row += _row_count(line)
    matrix = sparse.coo_matrix((values, (row_indices, columns)),
                               shape=(line_row, count)).tocsr()
    matrix.eliminate_zeros()
    return (matrix, tuple(samples), tuple(terms)) if with_terms else (matrix, tuple(samples))
