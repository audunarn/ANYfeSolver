"""Bounded segmented six-channel F03 weak-line work diagnostic.

This is an exploratory Lagrange-multiplier work integral, evaluated by
bounded Gauss quadrature over fixed reference correspondence. It composes
verified station ports and includes a small reference KKT oracle. It selects
neither a stable multiplier space nor a production attachment implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from anyfesolver._ge_beam3_pose_joint import RigidPoseJoint
from anyfesolver._ge_beam3_retained_generalized_state import Mechanical

from scripts._b3_e4_f03_coupled_point_work_candidate import (
    directional_point_work)
from scripts._b3_e4_f03_e4_trial_panel import (
    FlatE4TrialPanel, TrialPanelResponse)
from scripts._b3_e4_f03_native_fixture import Fixture, SECTION_DIAGONAL
from scripts._b3_e4_f03_native_spatial_port import NativeHalfCellSpatialPort
from scripts._b3_e4_f03_shell_finite_pose_candidate import pose_from_e4_total
from scripts._b3_e4_f03_shell_local_translation_candidate import (
    pose_from_e4_total_local_translation,
    pose_from_e4_total_local_director,
    pose_from_e4_total_local_frame,
    reference_spatial_trace_local_translation,
    reference_spatial_trace_local_director,
    reference_spatial_trace_local_frame)
from scripts._b3_e4_f03_shell_spatial_linear_port import reference_spatial_trace
from scripts._b3_e4_f03_shell_strip_pose_candidate import (
    LINE_ORIGIN_PORT, STRIP_PORTS, pose_from_e4_total_strip,
    reference_spatial_trace_strip, TRANSVERSE_BAND_PORT,
    pose_from_e4_total_transverse_band,
    reference_spatial_trace_transverse_band)
from scripts._b3_e4_f03_shell_chord_pose_candidate import (
    TRANSVERSE_TOP_CHORD_PORT, pose_from_e4_total_chord,
    reference_spatial_trace_chord)


@dataclass(frozen=True)
class LineMultiplierSpace:
    """P0, continuous P1 or local biorthogonal dual-P1 six-vectors."""

    kind: str
    knots: tuple[float, ...]

    def __post_init__(self) -> None:
        values = np.asarray(self.knots, dtype=float)
        if (self.kind not in ("piecewise_constant", "continuous_linear",
                              "dual_linear") or
                values.ndim != 1 or len(values) < 2 or
                not np.isfinite(values).all() or
                values[0] != 0. or values[-1] != 1. or
                not np.all(np.diff(values) > 0.)):
            raise ValueError("valid fixed line multiplier space required")
        object.__setattr__(self, "knots", tuple(float(value) for value in values))

    @property
    def count(self) -> int:
        return len(self.knots)-1 if self.kind == "piecewise_constant" else len(self.knots)

    def values(self, station: float) -> np.ndarray:
        if not np.isfinite(station) or not 0. <= station <= 1.:
            raise ValueError("finite line station in [0, 1] required")
        index = min(int(np.searchsorted(self.knots, station, side="right"))-1,
                    len(self.knots)-2)
        out = np.zeros(self.count)
        if self.kind == "piecewise_constant":
            out[index] = 1.
        else:
            fraction = ((station-self.knots[index]) /
                        (self.knots[index+1]-self.knots[index]))
            if self.kind == "continuous_linear":
                out[index:index+2] = (1.-fraction, fraction)
            else:
                # On one cell, psi_L=2*N_L-N_R and psi_R=2*N_R-N_L.
                # Assembly is biorthogonal to the continuous P1 nodal hats.
                out[index:index+2] = (2.-3.*fraction, 3.*fraction-1.)
        return out


@dataclass(frozen=True)
class WeakLineWork:
    potential: float
    weak_constraints: np.ndarray
    weak_first: np.ndarray
    weak_second: np.ndarray
    first: float
    second: float
    mixed: float
    segment_count: int
    quadrature_count: int
    shell_trial: TrialPanelResponse


@dataclass(frozen=True)
class ChannelSplitWeakWork:
    """Ordered coarse/fine channel work from one accepted predecessor."""

    potential: float
    weak_constraints: np.ndarray
    weak_first: np.ndarray
    weak_second: np.ndarray
    first: float
    second: float
    mixed: float
    segment_count: int
    quadrature_count: int
    shell_trial: TrialPanelResponse


@dataclass(frozen=True)
class ReferenceKKT:
    """Small dense oracle of the virgin constrained reference linearization."""

    primal_tangent: np.ndarray
    constraint_jacobian: np.ndarray
    free_primal: np.ndarray
    system: np.ndarray
    coordinate_scale: np.ndarray
    energy_scale: float
    scaled_system: np.ndarray
    segment_count: int
    shell_support_policy: str


def segments(shell: FlatE4TrialPanel, beam: Fixture,
             space: LineMultiplierSpace, *,
             shell_port: str = "rail") -> np.ndarray:
    """Split at shell hosts, native half-cells, and multiplier knots."""
    if not isinstance(shell, FlatE4TrialPanel) or not isinstance(beam, Fixture):
        raise TypeError("bounded E4 and native B3 fixtures required")
    shell.guard()
    beam.context.guard()
    if shell_port in (TRANSVERSE_BAND_PORT, TRANSVERSE_TOP_CHORD_PORT):
        x_grid = -.5+np.arange(shell.panel.x_interval_count+1)/shell.panel.n_s
        sites = list(x_grid)
        if shell.panel.topology == "S3-V2D":
            for y in (-.05, .05):
                eta = (y+.5)*shell.panel.n_y
                eta -= np.floor(eta)
                sites.extend(x_grid[:-1]+eta/shell.panel.n_s)
        shell_sites = np.array([s for s in sites if 0. <= s <= 1.])
    elif shell_port == "local_frame":
        shell_sites = -.5+np.arange(shell.panel.x_interval_count+1)/shell.panel.n_s
        shell_sites = shell_sites[(shell_sites >= 0.) & (shell_sites <= 1.)]
        shell_sites = np.r_[0., shell_sites, 1.]
    elif shell_port in STRIP_PORTS:
        half_length,half_width=STRIP_PORTS[shell_port]
        x_grid = -.5+np.arange(shell.panel.x_interval_count+1)/shell.panel.n_s
        sites = [0., 1.]
        if shell_port == LINE_ORIGIN_PORT:
            sites.extend(x_grid)
        for edge in x_grid:
            sites.extend((edge-half_length, edge+half_length))
        if shell.panel.topology == "S3-V2D":
            for y in (-half_width, half_width):
                eta = (y+.5)*shell.panel.n_y
                eta -= np.floor(eta)
                for edge in x_grid[:-1]:
                    diagonal = edge+eta/shell.panel.n_s
                    sites.extend((diagonal-half_length,
                                  diagonal+half_length))
        shell_sites = np.array([s for s in sites if 0. <= s <= 1.])
    elif shell_port in ("line_chord_020", "line_chord_confined_020"):
        x_grid = -.5+np.arange(shell.panel.x_interval_count+1)/shell.panel.n_s
        hosts = list(x_grid)
        if shell.panel.topology == "S3-V2D":
            eta = .5*shell.panel.n_y
            eta -= np.floor(eta)
            hosts.extend(x_grid[:-1]+eta/shell.panel.n_s)
        if shell_port == "line_chord_020":
            sites = [x-offset for x in hosts
                     for offset in (0., -.2, .2)]
        else:
            sites = [station for x in hosts for station in
                     (x, x/.6, (x-.4)/.6)]
        shell_sites = np.array([s for s in sites if 0. <= s <= 1.])
    elif shell_port in ("rail", "local_translation", "local_director"):
        shell_sites = shell.panel.breakpoints()
    else:
        raise ValueError("unknown bounded shell pose port")
    sites = np.r_[shell_sites,
                  np.arange(beam.n_half_cells+1)/beam.n_half_cells,
                  space.knots]
    edges = np.unique(np.round(sites, 14))
    if (edges[0] != 0. or edges[-1] != 1. or
            np.min(np.diff(edges)) <= 1e-12):
        raise ValueError("degenerate segmented line")
    return edges


def integrate_weak_line(
        shell: FlatE4TrialPanel, beam: Fixture,
        space: LineMultiplierSpace, multipliers: np.ndarray,
        shell_reference_total: np.ndarray, shell_total: np.ndarray,
        shell_first: np.ndarray, shell_second: np.ndarray,
        beam_state: Mechanical, beam_first: np.ndarray,
        beam_second: np.ndarray, *, order: int = 4,
        shell_port: str = "rail") -> WeakLineWork:
    """Integrate exact multiplier work and its physical directional Hessian.

    The reference joint is captured separately at every material station.
    The E4 material trial uses the same total coordinates as the shell pose.
    The native B3 state is a complete retained mechanical trial. No candidate
    from either domain is accepted by this diagnostic.
    """
    if not isinstance(space, LineMultiplierSpace):
        raise TypeError("line multiplier space required")
    if shell_port not in ("rail", "local_translation", "local_director",
                          "local_frame", "line_chord_020",
                          "line_chord_confined_020",
                          TRANSVERSE_TOP_CHORD_PORT, *STRIP_PORTS):
        raise ValueError("unknown bounded shell pose port")
    pose_map = {"rail": pose_from_e4_total,
                "local_translation": pose_from_e4_total_local_translation,
                "local_director": pose_from_e4_total_local_director,
                "local_frame": pose_from_e4_total_local_frame,
                **{name: lambda p, s, u, a=a, b=b, line=name==LINE_ORIGIN_PORT: (
                    pose_from_e4_total_strip(
                        p, s, u, half_length=a, half_width=b,
                        line_origin=line))
                   for name,(a,b) in STRIP_PORTS.items()},
                "line_chord_020": pose_from_e4_total_chord,
                TRANSVERSE_TOP_CHORD_PORT: lambda p, s, u: (
                    pose_from_e4_total_chord(
                        p, s, u, half_chord=.05,
                        axis="transverse_top")),
                "line_chord_confined_020": lambda p, s, u: (
                    pose_from_e4_total_chord(
                        p, s, u, confined=True))}[shell_port]
    if type(order) is not int or order not in (2, 3, 4, 5):
        raise ValueError("bounded Gauss order 2 through 5 required")
    coefficients = np.asarray(multipliers, dtype=float)
    if (coefficients.shape != (space.count, 6) or
            not np.isfinite(coefficients).all()):
        raise ValueError("finite six-vector multiplier coefficients required")
    edges = segments(shell, beam, space, shell_port=shell_port)
    # Both calls validate complete vectors and predecessor/definition identity.
    shell.evaluate(shell_reference_total, .5)
    shell_trial = shell.evaluate(shell_total, .5)
    points, weights = np.polynomial.legendre.leggauss(order)
    weak = np.zeros((space.count, 6))
    weak_first = np.zeros_like(weak)
    weak_second = np.zeros_like(weak)
    potential = first = second = mixed = 0.
    for left, right in zip(edges[:-1], edges[1:]):
        half = .5*(right-left)
        center = .5*(left+right)
        for point, gauss_weight in zip(points, weights):
            station = center+half*point
            basis = space.values(station)
            element, cell, fraction, _ = beam._locate(station)
            port = NativeHalfCellSpatialPort(
                beam.context, element, cell, fraction)
            shell_ref_position, shell_ref_frame = pose_map(
                shell.panel, station, shell_reference_total)
            beam_high, beam_low, beam_frame = port.pose_pair(
                beam.context.initial.mechanical)
            joint = RigidPoseJoint(
                np.array((shell_ref_position, beam_high+beam_low)),
                np.array((shell_ref_frame, beam_frame)))
            local = directional_point_work(
                joint, shell.panel, station, shell_total,
                shell_first, shell_second, port, beam_state,
                beam_first, beam_second, basis@coefficients,
                shell_port=shell_port)
            measure = half*gauss_weight
            potential += measure*local.potential
            weak += measure*basis[:, None]*local.constraints[None, :]
            weak_first += measure*basis[:, None]*local.constraints_first[None, :]
            weak_second += measure*basis[:, None]*local.constraints_second[None, :]
            first += measure*local.first
            second += measure*local.second
            mixed += measure*local.mixed
    return WeakLineWork(float(potential), weak, weak_first, weak_second,
                        float(first), float(second), float(mixed), len(edges)-1,
                        order*(len(edges)-1), shell_trial)


def assemble_reference_kkt(shell: FlatE4TrialPanel, beam: Fixture,
                           space: LineMultiplierSpace, *,
                           order: int = 4,
                           shell_port: str = "rail",
                           shell_support_policy: str = "legacy_three_left_nodes"
                           ) -> ReferenceKKT:
    """Assemble a bounded zero-multiplier shell/B3/weak-line KKT oracle.

    The shell and beam are at their flat virgin reference. The historical
    default fixes all six coordinates at three left-edge shell nodes.
    ``full_left_edge`` fixes all six coordinates at every left-edge node.
    ``both_xy_edges`` fixes in-plane coordinates at both complete x edges,
    plus out-of-plane coordinates on the left edge to remove rigid modes.
    This is a small full-matrix derivation check, not production assembly.
    """
    if not isinstance(space, LineMultiplierSpace):
        raise TypeError("line multiplier space required")
    if shell_port not in ("rail", "local_translation", "local_director",
                          "local_frame", "line_chord_020",
                          "line_chord_confined_020", TRANSVERSE_BAND_PORT,
                          TRANSVERSE_TOP_CHORD_PORT,
                          *STRIP_PORTS):
        raise ValueError("unknown bounded shell pose port")
    if shell_support_policy not in ("legacy_three_left_nodes",
                                    "full_left_edge", "both_xy_edges"):
        raise ValueError("unknown bounded shell support policy")
    pose_map = {"rail": pose_from_e4_total,
                "local_translation": pose_from_e4_total_local_translation,
                "local_director": pose_from_e4_total_local_director,
                 "local_frame": pose_from_e4_total_local_frame,
                 TRANSVERSE_BAND_PORT: pose_from_e4_total_transverse_band,
                 **{name: lambda p, s, u, a=a, b=b, line=name==LINE_ORIGIN_PORT: (
                     pose_from_e4_total_strip(
                         p, s, u, half_length=a, half_width=b,
                         line_origin=line))
                    for name,(a,b) in STRIP_PORTS.items()},
                "line_chord_020": pose_from_e4_total_chord,
                TRANSVERSE_TOP_CHORD_PORT: lambda p, s, u: (
                    pose_from_e4_total_chord(
                        p, s, u, half_chord=.05,
                        axis="transverse_top")),
                "line_chord_confined_020": lambda p, s, u: (
                    pose_from_e4_total_chord(
                        p, s, u, confined=True))}[shell_port]
    reference_trace = {
        "rail": reference_spatial_trace,
        "local_translation": reference_spatial_trace_local_translation,
        "local_director": reference_spatial_trace_local_director,
        "local_frame": reference_spatial_trace_local_frame,
        TRANSVERSE_BAND_PORT: reference_spatial_trace_transverse_band,
        **{name: lambda p, s, a=a, b=b, line=name==LINE_ORIGIN_PORT: (
            reference_spatial_trace_strip(
                p, s, half_length=a, half_width=b,
                line_origin=line))
           for name,(a,b) in STRIP_PORTS.items()},
        "line_chord_020": reference_spatial_trace_chord,
        TRANSVERSE_TOP_CHORD_PORT: lambda p, s: (
            reference_spatial_trace_chord(
                p, s, half_chord=.05,
                axis="transverse_top")),
        "line_chord_confined_020": lambda p, s: (
            reference_spatial_trace_chord(
                p, s, confined=True))}[shell_port]
    if type(order) is not int or order not in (2, 3, 4, 5):
        raise ValueError("bounded Gauss order 2 through 5 required")
    edges = segments(shell, beam, space, shell_port=shell_port)
    shell_size = 6*shell.panel.node_count
    beam_size = beam.context.count
    shell_trial = shell.evaluate(np.zeros(shell_size), .5)
    primal = np.zeros((shell_size+beam_size,)*2)
    primal[:shell_size, :shell_size] = shell_trial.tangent
    primal[shell_size:, shell_size:] = beam.tangent.toarray()
    constraint = np.zeros((6*space.count, shell_size+beam_size))
    points, weights = np.polynomial.legendre.leggauss(order)
    shell_zero = np.zeros(shell_size)
    for left, right in zip(edges[:-1], edges[1:]):
        half, center = .5*(right-left), .5*(left+right)
        for point, gauss_weight in zip(points, weights):
            station = center+half*point
            element, cell, fraction, _ = beam._locate(station)
            port = NativeHalfCellSpatialPort(
                beam.context, element, cell, fraction)
            shell_position, shell_frame = pose_map(
                shell.panel, station, shell_zero)
            beam_high, beam_low, beam_frame = port.pose_pair(
                beam.context.initial.mechanical)
            joint = RigidPoseJoint(
                np.array((shell_position, beam_high+beam_low)),
                np.array((shell_frame, beam_frame)))
            evaluation = joint.evaluate(
                np.array((shell_position, beam_high)),
                np.array((shell_frame, beam_frame)), np.zeros(6),
                position_low=np.array((np.zeros(3), beam_low)))
            if np.linalg.norm(evaluation.constraints) > 1e-11:
                raise ValueError("nonzero virgin weak-line reference gap")
            derivative = np.column_stack((
                evaluation.constraint_jacobian[:, :6]@
                reference_trace(shell.panel, station),
                evaluation.constraint_jacobian[:, 6:]@port.tangent(
                    beam.context.initial.mechanical)))
            for row, basis_value in enumerate(space.values(station)):
                constraint[6*row:6*row+6] += (
                    half*gauss_weight*basis_value*derivative)
    nx = 2*shell.panel.n_s
    if shell_support_policy == "legacy_three_left_nodes":
        clamped_shell_nodes = (0, nx+1, 2*(nx+1))
        clamped_shell = {6*node+axis for node in clamped_shell_nodes
                         for axis in range(6)}
    elif shell_support_policy == "full_left_edge":
        clamped_shell = {
            6*j*(nx+1)+axis
            for j in range(shell.panel.n_y+1)
            for axis in range(6)}
    else:
        clamped_shell = {
            6*(j*(nx+1)+i)+axis
            for j in range(shell.panel.n_y+1)
            for i in (0, nx) for axis in (0, 1)}
        clamped_shell.update({
            6*(j*(nx+1))+axis
            for j in range(shell.panel.n_y+1)
            for axis in (2, 3, 4)})
    fixed_beam = {shell_size+index for index in beam.context.fixed}
    free = np.array([index for index in range(shell_size+beam_size)
                     if index not in clamped_shell and index not in fixed_beam],
                    dtype=int)
    reduced = primal[np.ix_(free, free)]
    rows = constraint[:, free]
    system = np.block([[reduced, rows.T],
                       [rows, np.zeros((len(rows), len(rows)))]])
    force_scale = 1e6
    length_scale = 1.
    displacement_scale = force_scale/(210e9*.07)
    nodal_scale = np.tile(np.r_[np.full(3, displacement_scale),
                                np.full(3, displacement_scale/length_scale)],
                          shell.panel.node_count+beam.context.nodal_count//6)
    resultants = np.r_[np.tile(SECTION_DIAGONAL[:3]*displacement_scale, 2),
                       np.tile(SECTION_DIAGONAL[3:]*
                               (displacement_scale/length_scale), 4)]
    retained_scale = np.tile(np.r_[
        np.full(6, displacement_scale/length_scale), resultants],
        beam.n_half_cells//2)
    full_scale = np.r_[nodal_scale, retained_scale]
    if len(full_scale) != shell_size+beam_size:
        raise ValueError("native retained scaling layout changed")
    coordinate_scale = np.r_[full_scale[free],
                             np.full(len(rows), force_scale)]
    energy_scale = force_scale*displacement_scale
    scaled_system = ((coordinate_scale[:, None]*system)*
                     coordinate_scale[None, :])/energy_scale
    return ReferenceKKT(primal, constraint, free, system,
                        coordinate_scale, energy_scale,
                        scaled_system, len(edges)-1, shell_support_policy)


def _channel_split_rows(coarse_intervals: int, fine_intervals: int,
                        fine_channels: tuple[int, ...]
                        ) -> tuple[list[int], list[int]]:
    """Validate and number each physical channel exactly once."""
    if (type(coarse_intervals) is not int or
            type(fine_intervals) is not int or
            not 1 <= coarse_intervals <= fine_intervals <= 16 or
            fine_intervals % coarse_intervals):
        raise ValueError("bounded nested P0 interval counts required")
    if (type(fine_channels) is not tuple or
            any(type(channel) is not int or not 0 <= channel < 6
                for channel in fine_channels) or
            tuple(sorted(set(fine_channels))) != fine_channels):
        raise ValueError("ordered unique physical channel indices required")
    coarse_rows=[6*i+j for i in range(coarse_intervals)
                 for j in range(6) if j not in fine_channels]
    fine_rows=[6*i+j for i in range(fine_intervals)
               for j in fine_channels]
    return coarse_rows,fine_rows


def assemble_channel_split_reference_kkt(
        shell: FlatE4TrialPanel, beam: Fixture, *,
        coarse_intervals: int, fine_intervals: int,
        fine_channels: tuple[int, ...],
        shell_port: str = "physical_strip_005_005",
        shell_support_policy: str = "full_left_edge") -> ReferenceKKT:
    """Bounded P0 work pairing with explicit per-channel resolution.

    Each channel is taken from exactly one independently integrated
    reference weak line. No multiplier row is averaged, penalized or
    duplicated. This is a G0 diagnostic candidate, not a space selection.
    """
    coarse_rows,fine_rows=_channel_split_rows(
        coarse_intervals,fine_intervals,fine_channels)

    def assemble(intervals: int) -> ReferenceKKT:
        space=LineMultiplierSpace(
            "piecewise_constant",tuple(np.linspace(0.,1.,intervals+1)))
        return assemble_reference_kkt(
            shell,beam,space,shell_port=shell_port,
            shell_support_policy=shell_support_policy)

    coarse=assemble(coarse_intervals)
    fine=coarse if fine_intervals==coarse_intervals else assemble(fine_intervals)
    n=len(coarse.free_primal)
    if (not np.array_equal(coarse.free_primal,fine.free_primal) or
            not np.array_equal(coarse.primal_tangent,fine.primal_tangent) or
            not np.array_equal(coarse.coordinate_scale[:n],
                               fine.coordinate_scale[:n]) or
            coarse.energy_scale != fine.energy_scale):
        raise ArithmeticError("split channel primal contracts disagree")
    constraint=np.vstack((coarse.constraint_jacobian[coarse_rows],
                          fine.constraint_jacobian[fine_rows])).copy()
    rows=constraint[:,coarse.free_primal]
    channels=[j%6 for j in coarse_rows+fine_rows]
    row_scale=np.array([1. if j<3 else .46 for j in channels])
    scaled_rows=(row_scale[:,None]*rows)*coarse.coordinate_scale[None,:n]
    if np.linalg.matrix_rank(scaled_rows) != len(rows):
        raise ValueError("split channel rows are algebraically dependent")
    reduced=coarse.primal_tangent[np.ix_(coarse.free_primal,
                                         coarse.free_primal)]
    system=np.block([[reduced,rows.T],
                     [rows,np.zeros((len(rows),len(rows)))]])
    multiplier_scale=coarse.coordinate_scale[n]
    coordinate_scale=np.r_[coarse.coordinate_scale[:n],
                           np.full(len(rows),multiplier_scale)]
    scaled_system=((coordinate_scale[:,None]*system)*
                   coordinate_scale[None,:])/coarse.energy_scale
    return ReferenceKKT(coarse.primal_tangent.copy(),constraint,
                        coarse.free_primal.copy(),system,coordinate_scale,
                        coarse.energy_scale,scaled_system,
                        max(coarse.segment_count,fine.segment_count),
                        shell_support_policy)


def integrate_channel_split_weak_line(
        shell: FlatE4TrialPanel, beam: Fixture,
        coarse_intervals: int, fine_intervals: int,
        fine_channels: tuple[int, ...], multipliers: np.ndarray,
        shell_reference_total: np.ndarray, shell_total: np.ndarray,
        shell_first: np.ndarray, shell_second: np.ndarray,
        beam_state: Mechanical, beam_first: np.ndarray,
        beam_second: np.ndarray, *, order: int = 4,
        shell_port: str = "physical_strip_005_005") -> ChannelSplitWeakWork:
    """Compose exact nonlinear weak work with channel-specific P0 tests.

    Both integrals evaluate the same physical trial and accepted predecessor.
    Their selected rows form one multiplier potential; no domain commits.
    """
    coarse_rows,fine_rows=_channel_split_rows(
        coarse_intervals,fine_intervals,fine_channels)
    coefficients=np.asarray(multipliers,dtype=float)
    if (coefficients.shape != (len(coarse_rows)+len(fine_rows),) or
            not np.isfinite(coefficients).all()):
        raise ValueError("finite split-channel multipliers required")
    coarse_coeff=np.zeros((coarse_intervals,6))
    fine_coeff=np.zeros((fine_intervals,6))
    coarse_coeff.ravel()[coarse_rows]=coefficients[:len(coarse_rows)]
    fine_coeff.ravel()[fine_rows]=coefficients[len(coarse_rows):]
    arguments=(shell_reference_total,shell_total,shell_first,shell_second,
               beam_state,beam_first,beam_second)
    coarse=integrate_weak_line(
        shell,beam,LineMultiplierSpace("piecewise_constant",
            tuple(np.linspace(0.,1.,coarse_intervals+1))),
        coarse_coeff,*arguments,order=order,shell_port=shell_port)
    fine=integrate_weak_line(
        shell,beam,LineMultiplierSpace("piecewise_constant",
            tuple(np.linspace(0.,1.,fine_intervals+1))),
        fine_coeff,*arguments,order=order,shell_port=shell_port)
    def selected(name: str) -> np.ndarray:
        return np.r_[getattr(coarse,name).ravel()[coarse_rows],
                     getattr(fine,name).ravel()[fine_rows]].copy()
    result=ChannelSplitWeakWork(
        coarse.potential+fine.potential,
        selected("weak_constraints"),selected("weak_first"),
        selected("weak_second"),coarse.first+fine.first,
        coarse.second+fine.second,coarse.mixed+fine.mixed,
        coarse.segment_count+fine.segment_count,
        coarse.quadrature_count+fine.quadrature_count,
        coarse.shell_trial)
    if abs(result.potential-coefficients@result.weak_constraints) > (
            1e-10*max(1.,abs(result.potential))):
        raise ArithmeticError("split-channel potential/weak-row mismatch")
    return result


def _bounded_channel_spaces(
        spaces: tuple[LineMultiplierSpace, ...]) -> None:
    """Validate six explicitly chosen physical multiplier test spaces."""
    if (type(spaces) is not tuple or len(spaces) != 6 or
            any(not isinstance(space, LineMultiplierSpace)
                for space in spaces) or
            any(space.count > 17 for space in spaces) or
            sum(space.count for space in spaces) > 96):
        raise ValueError("six bounded physical channel spaces required")


def assemble_channel_basis_reference_kkt(
        shell: FlatE4TrialPanel, beam: Fixture,
        spaces: tuple[LineMultiplierSpace, ...], *,
        shell_port: str = "physical_strip_005_005",
        shell_support_policy: str = "full_left_edge") -> ReferenceKKT:
    """Exact reference rows for independently chosen channel bases.

    Rows are grouped Fx,Fy,Fz,Mx,My,Mz, with each group's own basis order.
    This bounded G0 oracle neither selects a multiplier nor condenses the
    native beam's retained coordinates.
    """
    _bounded_channel_spaces(spaces)
    cache: dict[LineMultiplierSpace, ReferenceKKT] = {}
    for space in spaces:
        if space not in cache:
            cache[space] = assemble_reference_kkt(
                shell, beam, space, shell_port=shell_port,
                shell_support_policy=shell_support_policy)
    reference = cache[spaces[0]]
    n = len(reference.free_primal)
    if n+sum(space.count for space in spaces) > 1200:
        raise ValueError("bounded dense interface diagnostic exceeds 1200 rows")
    for candidate in cache.values():
        if (not np.array_equal(reference.free_primal,
                               candidate.free_primal) or
                not np.array_equal(reference.primal_tangent,
                                   candidate.primal_tangent) or
                not np.array_equal(reference.coordinate_scale[:n],
                                   candidate.coordinate_scale[:n]) or
                reference.energy_scale != candidate.energy_scale):
            raise ArithmeticError("channel basis primal contracts disagree")
    constraint = np.vstack([
        cache[space].constraint_jacobian[axis::6]
        for axis, space in enumerate(spaces)]).copy()
    rows = constraint[:, reference.free_primal]
    channel_ids = [axis for axis, space in enumerate(spaces)
                   for _ in range(space.count)]
    row_scale = np.array([1. if axis < 3 else .46
                          for axis in channel_ids])
    scaled_rows = ((row_scale[:, None]*rows)*
                   reference.coordinate_scale[None, :n])
    if np.linalg.matrix_rank(scaled_rows) != len(rows):
        raise ValueError("channel basis rows are algebraically dependent")
    k = reference.primal_tangent[np.ix_(reference.free_primal,
                                        reference.free_primal)]
    system = np.block([[k, rows.T],
                       [rows, np.zeros((len(rows), len(rows)))]])
    multiplier_scale = reference.coordinate_scale[n]
    coordinate_scale = np.r_[reference.coordinate_scale[:n],
                             np.full(len(rows), multiplier_scale)]
    scaled_system = ((coordinate_scale[:, None]*system)*
                     coordinate_scale[None, :])/reference.energy_scale
    return ReferenceKKT(
        reference.primal_tangent.copy(), constraint,
        reference.free_primal.copy(), system, coordinate_scale,
        reference.energy_scale, scaled_system,
        max(candidate.segment_count for candidate in cache.values()),
        shell_support_policy)


def integrate_channel_basis_weak_line(
        shell: FlatE4TrialPanel, beam: Fixture,
        spaces: tuple[LineMultiplierSpace, ...],
        multipliers: np.ndarray,
        shell_reference_total: np.ndarray, shell_total: np.ndarray,
        shell_first: np.ndarray, shell_second: np.ndarray,
        beam_state: Mechanical, beam_first: np.ndarray,
        beam_second: np.ndarray, *, order: int = 4,
        shell_port: str = "physical_strip_005_005") -> ChannelSplitWeakWork:
    """One exact multiplier potential assembled from six channel bases."""
    _bounded_channel_spaces(spaces)
    coefficients = np.asarray(multipliers, dtype=float)
    if (coefficients.shape != (sum(space.count for space in spaces),) or
            not np.isfinite(coefficients).all()):
        raise ValueError("finite channel-basis multiplier vector required")
    arguments = (shell_reference_total, shell_total, shell_first,
                 shell_second, beam_state, beam_first, beam_second)
    cache: dict[LineMultiplierSpace, np.ndarray] = {}
    offset = 0
    for axis, space in enumerate(spaces):
        if space not in cache:
            cache[space] = np.zeros((space.count, 6))
        cache[space][:, axis] = coefficients[offset:offset+space.count]
        offset += space.count
    evaluated = {space: integrate_weak_line(
        shell, beam, space, grouped, *arguments, order=order,
        shell_port=shell_port) for space, grouped in cache.items()}
    selected = lambda name: np.concatenate([
        getattr(evaluated[space], name)[:, axis]
        for axis, space in enumerate(spaces)]).copy()
    first = next(iter(evaluated.values()))
    result = ChannelSplitWeakWork(
        sum(work.potential for work in evaluated.values()),
        selected("weak_constraints"), selected("weak_first"),
        selected("weak_second"),
        sum(work.first for work in evaluated.values()),
        sum(work.second for work in evaluated.values()),
        sum(work.mixed for work in evaluated.values()),
        sum(work.segment_count for work in evaluated.values()),
        sum(work.quadrature_count for work in evaluated.values()),
        first.shell_trial)
    if abs(result.potential-coefficients@result.weak_constraints) > (
            1e-10*max(1., abs(result.potential))):
        raise ArithmeticError("channel-basis potential/weak-row mismatch")
    return result
