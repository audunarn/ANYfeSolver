"""Unregistered small-system F03 multiplier-resolution diagnostic.

This reuses the bounded native B3/E4 reference KKT oracle. It compares
multiplier spaces on identical physical meshes; its pointwise gaps and
linearized responses are descriptive, not a stability or accuracy gate.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy import linalg

from anyfesolver._ge_beam3_pose_joint import RigidPoseJoint

from scripts._b3_e4_f03_e4_trial_panel import FlatE4TrialPanel
from scripts._b3_e4_f03_native_fixture import build
from scripts._b3_e4_f03_native_spatial_port import NativeHalfCellSpatialPort
from scripts._b3_e4_f03_shell_finite_pose_candidate import (
    pose_from_e4_total, reference_nodes)
from scripts._b3_e4_f03_shell_local_translation_candidate import (
    pose_from_e4_total_local_translation,
    pose_from_e4_total_local_director,
    pose_from_e4_total_local_frame)
from scripts._b3_e4_f03_shell_strip_pose_candidate import (
    LINE_ORIGIN_PORT, STRIP_PORTS, pose_from_e4_total_strip,
    TRANSVERSE_BAND_PORT, pose_from_e4_total_transverse_band)
from scripts._b3_e4_f03_shell_chord_pose_candidate import (
    TRANSVERSE_TOP_CHORD_PORT, pose_from_e4_total_chord)
from scripts._b3_e4_f03_shell_spatial_linear_port import physical_shell_stiffness
from scripts._b3_e4_f03_shell_trace import make_shell
from scripts._b3_e4_f03_weak_line_candidate import (
    LineMultiplierSpace, assemble_channel_basis_reference_kkt,
    assemble_channel_split_reference_kkt,
    assemble_reference_kkt)
from scripts._b3_e4_f03_p0_dual_norm import p0_dual_gram
from scripts._b3_e4_f03_p1_dual_norm import p1_dual_gram
from scripts._b3_e4_f03_dual_linear_norm import (
    dual_linear_gram, dual_linear_l2_gram)


def sinusoidal_axial_density_load(beam, *, amplitude_n_per_m=1e5):
    """Integrate one fixed physical line density through native station work."""
    if (isinstance(amplitude_n_per_m, (bool, str)) or
            not isinstance(amplitude_n_per_m,
                           (int, float, np.integer, np.floating)) or
            not np.isfinite(amplitude_n_per_m) or
            amplitude_n_per_m <= 0.):
        raise ValueError("positive finite axial line-load amplitude required")
    points, weights = np.polynomial.legendre.leggauss(8)
    load = np.zeros(beam.context.count)
    for half_cell in range(beam.n_half_cells):
        left = half_cell/beam.n_half_cells
        right = (half_cell+1)/beam.n_half_cells
        for point, weight in zip(points, weights):
            station = (left+right)/2+(right-left)*point/2
            element, cell, fraction, _ = beam._locate(float(station))
            port = NativeHalfCellSpatialPort(
                beam.context, element, cell, fraction)
            wrench = np.array((amplitude_n_per_m*np.sin(2*np.pi*station),
                               0., 0., 0., 0., 0.))
            load += (right-left)*weight/2*port.pullback(
                beam.context.initial.mechanical, wrench)
    return load


def one_case(topology: str, intervals: int, kind: str, *,
             shell_port: str = "rail",
             load_kind: str = "torsion", n_s: int = 4,
             native_half_cells: int = 16,
             n_y: int = 2,
             thickness: float = .07,
             beam_z_offset: float = 0.,
             beam_torsion_constant: float | None = None,
             shell_support_policy: str = "legacy_three_left_nodes",
             fine_intervals: int | None = None,
             fine_channels: tuple[int, ...] = (),
             channel_spaces: tuple[LineMultiplierSpace, ...] | None = None
             ) -> dict:
    """Solve one bounded six-channel load at fixed reference resolution."""
    shell = FlatE4TrialPanel(make_shell(n_s, topology, n_y=n_y,
                                       thickness=thickness))
    beam = build(native_half_cells, z_offset=beam_z_offset,
                 torsion_constant=beam_torsion_constant)
    space = LineMultiplierSpace(
        kind, tuple(np.linspace(0., 1., intervals+1)))
    if channel_spaces is not None:
        if fine_intervals is not None or fine_channels:
            raise ValueError("explicit channel spaces cannot combine with split intervals")
        kkt = assemble_channel_basis_reference_kkt(
            shell, beam, channel_spaces, shell_port=shell_port,
            shell_support_policy=shell_support_policy)
    elif fine_intervals is None:
        if fine_channels:
            raise ValueError("fine channels require a fine interval count")
        kkt = assemble_reference_kkt(
            shell, beam, space, shell_port=shell_port,
            shell_support_policy=shell_support_policy)
    else:
        if kind != "piecewise_constant":
            raise ValueError("split channels require P0 multipliers")
        kkt = assemble_channel_split_reference_kkt(
            shell, beam, coarse_intervals=intervals,
            fine_intervals=fine_intervals,fine_channels=fine_channels,
            shell_port=shell_port,
            shell_support_policy=shell_support_policy)
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
    shell_size = 6*shell.panel.node_count
    tip_rx = shell_size+6*beam.n_half_cells+3
    tip_rz = shell_size+6*beam.n_half_cells+5
    tip_uz = shell_size+6*beam.n_half_cells+2
    load = np.zeros(shell_size+beam.context.count)
    if load_kind == "torsion":
        load[tip_rx] = 100.
    elif load_kind == "yaw_torque_z":
        load[tip_rz] = 100.
    elif load_kind == "shell_force_couple_x":
        sample = shell.panel.sample_weights(1.)
        y = np.array([-.23, -.23, .23, .23])
        load[:shell_size].reshape(-1, 6)[:, 2] += (
            100.*(y/(y@y))@sample)
    elif load_kind == "transverse_force_z":
        load[tip_uz] = 1000.
    elif load_kind == "axial_wave":
        positions = (beam.context.initial.mechanical.positions+
                     beam.context.initial.mechanical.position_low)
        axial = 1e5*np.sin(8*np.pi*positions[:, 0])
        axial[np.abs(axial) < 1e-8] = 0.
        load[shell_size:shell_size+beam.context.nodal_count:6] = axial
    elif load_kind == "sinusoidal_axial_density":
        load[shell_size:] = sinusoidal_axial_density_load(beam)
    else:
        raise ValueError("unknown bounded line load kind")
    free_count = len(kkt.free_primal)
    rhs = np.r_[load[kkt.free_primal]*
                kkt.coordinate_scale[:free_count]/kkt.energy_scale,
                np.zeros(kkt.constraint_jacobian.shape[0])]
    scaled_solution = np.linalg.solve(kkt.scaled_system, rhs)
    scaled_residual = kkt.scaled_system@scaled_solution-rhs
    physical_solution = kkt.coordinate_scale*scaled_solution
    full = np.zeros(len(load))
    full[kkt.free_primal] = physical_solution[:free_count]
    multipliers = physical_solution[free_count:]
    physical_residual = (kkt.primal_tangent@full+
                         kkt.constraint_jacobian.T@multipliers-load)
    shell_fixed = np.ones(shell_size, dtype=bool)
    shell_fixed[kkt.free_primal[kkt.free_primal < shell_size]] = False
    shell_reaction_fx = float(np.sum(
        physical_residual[:shell_size:6][shell_fixed[0::6]]))
    beam_root_reaction_fx = float(physical_residual[shell_size])
    shell_support = physical_residual[:shell_size].copy()
    shell_support[~shell_fixed] = 0.
    shell_support = shell_support.reshape(-1, 6)
    shell_reference_nodes = reference_nodes(shell.panel)
    shell_interface = (kkt.constraint_jacobian[:, :shell_size].T@
                       multipliers).reshape(-1, 6)
    shell_interface_torque_x = float(np.sum(
        shell_interface[:, 3]+shell_reference_nodes[:, 1]*shell_interface[:, 2]
        -shell_reference_nodes[:, 2]*shell_interface[:, 1]))
    shell_interface_force_z = float(np.sum(shell_interface[:, 2]))
    shell_interface_mz = float(np.sum(
        shell_interface[:, 5]+shell_reference_nodes[:, 0]*shell_interface[:, 1]
        -shell_reference_nodes[:, 1]*shell_interface[:, 0]))
    shell_reaction_mx = float(np.sum(
        shell_support[:, 3]+shell_reference_nodes[:, 1]*shell_support[:, 2]
        -shell_reference_nodes[:, 2]*shell_support[:, 1]))
    shell_reaction_fz = float(np.sum(shell_support[:, 2]))
    shell_reaction_my = float(np.sum(
        shell_support[:, 4]+shell_reference_nodes[:, 2]*shell_support[:, 0]
        -shell_reference_nodes[:, 0]*shell_support[:, 2]))
    shell_reaction_mz = float(np.sum(
        shell_support[:, 5]+shell_reference_nodes[:, 0]*shell_support[:, 1]
        -shell_reference_nodes[:, 1]*shell_support[:, 0]))
    beam_root_reaction_mx = float(physical_residual[shell_size+3])
    beam_root_reaction_fy = float(physical_residual[shell_size+1])
    beam_root_reaction_mx_about_origin = float(
        beam_root_reaction_mx-beam_z_offset*beam_root_reaction_fy)
    beam_root_reaction_fz = float(physical_residual[shell_size+2])
    beam_root_reaction_my = float(physical_residual[shell_size+4])
    beam_root_reaction_mz = float(physical_residual[shell_size+5])
    shell_load = load[:shell_size].reshape(-1, 6)
    applied_torque_x = float(load[tip_rx]+np.sum(
        shell_load[:, 3]+shell_reference_nodes[:, 1]*shell_load[:, 2]
        -shell_reference_nodes[:, 2]*shell_load[:, 1]))
    applied_force_z = float(load[tip_uz]+np.sum(shell_load[:, 2]))
    applied_moment_y = float(-load[tip_uz]+np.sum(
        shell_load[:, 4]+shell_reference_nodes[:, 2]*shell_load[:, 0]
        -shell_reference_nodes[:, 0]*shell_load[:, 2]))
    applied_moment_z = float(load[tip_rz]+np.sum(
        shell_load[:, 5]+shell_reference_nodes[:, 0]*shell_load[:, 1]
        -shell_reference_nodes[:, 1]*shell_load[:, 0]))
    shell_total = full[:shell_size]
    sample_z = shell.panel.sample_weights(1.)@shell_total.reshape(-1, 6)[:, 2]
    sample_y = np.array([-.23, -.23, .23, .23])
    shell_tip_z_slope = float(sample_y@sample_z/(sample_y@sample_y))
    shell_inplane = shell_total.reshape(-1, 6)[:, :2].ravel()
    shell_physical_inplane_energy = float(
        .5*shell_inplane@(shell.panel.stiffness@ shell_inplane))
    shell_full_energy = float(.5*shell_total@kkt.primal_tangent[
        :shell_size, :shell_size]@shell_total)
    shell_physical_energy = float(
        .5*shell_total@(physical_shell_stiffness(shell.panel)@shell_total))
    shell_component_physical_energy = 0.
    shell_numerical_energy = 0.
    for _, nodes, element in shell.elements:
        components = element.compute_stiffness_components(
            shell.model.mesh, shell.model.get_material("steel"))
        local = shell_total.reshape(-1, 6)[np.asarray(nodes)-1].ravel()
        shell_component_physical_energy += float(
            .5*local@components["physical"]@local)
        shell_numerical_energy += float(
            .5*local@components["numerical"]@local)
    beam_energy = float(.5*full[shell_size:]@
                        kkt.primal_tangent[shell_size:, shell_size:]@
                        full[shell_size:])
    beam_state = beam.context.advance(
        beam.context.initial.mechanical, full[shell_size:])
    shell_reference = np.zeros(shell_size)
    gap_norms = []
    axial_gaps = []
    for station in np.linspace(0., 1., 129):
        element, cell, fraction, _ = beam._locate(float(station))
        port = NativeHalfCellSpatialPort(
            beam.context, element, cell, fraction)
        shell_ref_position, shell_ref_frame = pose_map(
            shell.panel, float(station), shell_reference)
        beam_ref_high, beam_ref_low, beam_ref_frame = port.pose_pair(
            beam.context.initial.mechanical)
        joint = RigidPoseJoint(
            np.array((shell_ref_position, beam_ref_high+beam_ref_low)),
            np.array((shell_ref_frame, beam_ref_frame)))
        shell_position, shell_frame = pose_map(
            shell.panel, float(station), shell_total)
        beam_high, beam_low, beam_frame = port.pose_pair(beam_state)
        gap = joint.evaluate(
            np.array((shell_position, beam_high)),
            np.array((shell_frame, beam_frame)), np.zeros(6),
            position_low=np.array((np.zeros(3), beam_low))).constraints
        gap_norms.append(float(np.linalg.norm(np.r_[gap[:3], gap[3:]*1.])))
        axial_gaps.append(float(gap[0]))
    gaps = np.asarray(gap_norms)
    axial_gap = np.asarray(axial_gaps)
    beam_nodal = full[
        shell_size:shell_size+beam.context.nodal_count].reshape(-1, 6)
    return {
        "topology": topology,
        "kind": "channel_specific" if channel_spaces is not None else kind,
        "shell_port": shell_port,
        "shell_support_policy": shell_support_policy,
        "load_kind": load_kind,
        "intervals": intervals,
        "fine_intervals":fine_intervals,
        "fine_channels":fine_channels,
        "channel_spaces":(None if channel_spaces is None else tuple(
            {"kind": entry.kind, "knots": entry.knots}
            for entry in channel_spaces)),
        "shell_host_intervals": n_s,
        "shell_transverse_intervals": n_y,
        "shell_thickness_m": shell.panel.thickness,
        "beam_reference_z_m": beam_z_offset,
        "beam_torsion_constant_m4": beam_torsion_constant,
        "native_half_cells": native_half_cells,
        "shell_elements": shell.panel.element_count,
        "native_elements": beam.n_half_cells//2,
        "tip_rx_rad": float(full[tip_rx]),
        "tip_rz_rad": float(full[tip_rz]),
        "shell_tip_z_antisymmetric_slope_rad": shell_tip_z_slope,
        "beam_tip_uz_m": float(full[tip_uz]),
        "beam_tip_ux_m": float(beam_nodal[-1, 0]),
        "beam_max_abs_ux_m": float(np.max(np.abs(beam_nodal[:, 0]))),
        "shell_max_abs_ux_m": float(np.max(np.abs(
            full[:shell_size].reshape(-1, 6)[:, 0]))),
        "applied_axial_force_n": float(np.sum(
            load[shell_size:shell_size+beam.context.nodal_count:6])),
        "applied_axial_first_moment_n_m": float(np.dot(
            np.arange(native_half_cells+1)/native_half_cells,
            load[shell_size:shell_size+beam.context.nodal_count:6])),
        "shell_energy_j": shell_full_energy,
        "shell_physical_energy_j": shell_physical_energy,
        "shell_component_physical_energy_j": shell_component_physical_energy,
        "shell_numerical_energy_j": shell_numerical_energy,
        "shell_component_closure_error_j": float(
            shell_full_energy-shell_component_physical_energy-
            shell_numerical_energy),
        "shell_numerical_energy_fraction": (
            shell_numerical_energy/shell_full_energy
            if shell_full_energy != 0. else None),
        "shell_physical_inplane_energy_j": shell_physical_inplane_energy,
        "shell_full_minus_physical_inplane_energy_j": float(
            shell_full_energy-shell_physical_inplane_energy),
        "beam_energy_j": beam_energy,
        "global_work_error_j": float(
            2*(shell_full_energy+beam_energy)-full@load),
        "shell_reaction_fx_n": shell_reaction_fx,
        "beam_root_reaction_fx_n": beam_root_reaction_fx,
        "global_axial_balance_n": float(
            shell_reaction_fx+beam_root_reaction_fx+
            np.sum(load[shell_size:shell_size+beam.context.nodal_count:6])),
        "applied_torque_x_n_m": applied_torque_x,
        "shell_support_reaction_torque_x_n_m": shell_reaction_mx,
        "shell_interface_torque_x_n_m": shell_interface_torque_x,
        "shell_support_minus_interface_torque_x_n_m": float(
            shell_reaction_mx-shell_interface_torque_x),
        "beam_root_reaction_torque_x_n_m": beam_root_reaction_mx,
        "beam_root_reaction_fy_n": beam_root_reaction_fy,
        "beam_root_reaction_torque_x_about_origin_n_m": (
            beam_root_reaction_mx_about_origin),
        "global_torque_x_balance_n_m": float(
            shell_reaction_mx+beam_root_reaction_mx_about_origin+
            applied_torque_x),
        "applied_force_z_n": applied_force_z,
        "shell_support_reaction_fz_n": shell_reaction_fz,
        "shell_interface_fz_n": shell_interface_force_z,
        "shell_support_minus_interface_fz_n": float(
            shell_reaction_fz-shell_interface_force_z),
        "beam_root_reaction_fz_n": beam_root_reaction_fz,
        "global_force_z_balance_n": float(
            shell_reaction_fz+beam_root_reaction_fz+applied_force_z),
        "applied_moment_y_about_origin_n_m": applied_moment_y,
        "shell_support_reaction_my_about_origin_n_m": shell_reaction_my,
        "beam_root_reaction_my_about_origin_n_m": beam_root_reaction_my,
        "global_moment_y_balance_n_m": float(
            shell_reaction_my+beam_root_reaction_my+applied_moment_y),
        "applied_moment_z_about_origin_n_m": applied_moment_z,
        "shell_support_reaction_mz_about_origin_n_m": shell_reaction_mz,
        "shell_interface_mz_about_origin_n_m": shell_interface_mz,
        "beam_root_reaction_mz_about_origin_n_m": beam_root_reaction_mz,
        "global_moment_z_balance_n_m": float(
            shell_reaction_mz+beam_root_reaction_mz+applied_moment_z),
        "gap_l2_m": float(np.sqrt(np.trapezoid(gaps*gaps, dx=1./128))),
        "gap_max_m": float(np.max(gaps)),
        "gap_axial_l2_m": float(np.sqrt(np.trapezoid(
            axial_gap*axial_gap, dx=1./128))),
        "gap_axial_max_m": float(np.max(np.abs(axial_gap))),
        "scaled_equilibrium_residual": float(np.linalg.norm(scaled_residual)),
        "free_primal_residual_mixed_l2": float(np.linalg.norm(
            physical_residual[kkt.free_primal])),
    }


def chord_reference_modes(topology: str, intervals: int, kind: str, *,
                          n_s: int = 4, n_y: int = 4,
                          native_half_cells: int = 8,
                          characteristic_length: float = .46,
                          confined: bool = False,
                          shell_port: str | None = None,
                          loaded_mode_overlap: bool = False,
                          thickness: float = .07,
                          beam_z_offset: float = 0.,
                          beam_torsion_constant: float | None = None) -> dict:
    """Bounded six-channel, work-scaled interface diagnostic.

    Exact elimination of the indefinite native retained block separates a
    positive physical stiffness from signed local retained work. The norm
    uses physical energy plus the absolute retained quadratic form; it is
    an algorithmic finite-mesh diagnostic, not a physical continuum inf-sup
    proof or multiplier-space selection. E4 numerical completion remains in
    the stiffness and must be assessed separately.
    """
    if not np.isfinite(characteristic_length) or characteristic_length <= 0.:
        raise ValueError("positive physical characteristic length required")
    if type(loaded_mode_overlap) is not bool:
        raise ValueError("Boolean loaded-mode diagnostic required")
    shell = FlatE4TrialPanel(make_shell(n_s, topology, n_y=n_y,
                                       thickness=thickness))
    beam = build(native_half_cells, z_offset=beam_z_offset,
                 torsion_constant=beam_torsion_constant)
    space = LineMultiplierSpace(
        kind, tuple(np.linspace(0., 1., intervals+1)))
    if type(confined) is not bool:
        raise ValueError("Boolean attachment confinement required")
    if shell_port is None:
        shell_port=("line_chord_confined_020" if confined else
                    "line_chord_020")
    elif confined:
        raise ValueError("explicit shell port cannot combine with confinement")
    kkt = assemble_reference_kkt(
        shell, beam, space, shell_port=shell_port,
        shell_support_policy="full_left_edge")
    free = kkt.free_primal
    rows = kkt.constraint_jacobian[:, free]
    n = len(free)
    if n+len(rows) > 1200:
        raise ValueError("bounded dense interface diagnostic exceeds 1200 rows")
    n_physical = int(np.count_nonzero(
        free < 6*shell.panel.node_count+beam.context.nodal_count))
    displacement_scale = 1e6/(210e9*.07)
    if not np.isclose(kkt.energy_scale, 1e6*displacement_scale,
                      rtol=1e-14, atol=0.):
        raise ValueError("reference KKT work scale changed")
    scales = kkt.coordinate_scale[:n]
    k = kkt.primal_tangent[np.ix_(free, free)]
    kbar = (scales[:, None]*k)*scales[None, :]/kkt.energy_scale
    if np.linalg.norm(kbar-kbar.T) > 1e-10*np.linalg.norm(kbar):
        raise ValueError("nonsymmetric reference tangent cannot use energy norm")
    kpp = kbar[:n_physical, :n_physical]
    kpr = kbar[:n_physical, n_physical:]
    krp = kbar[n_physical:, :n_physical]
    krr = kbar[n_physical:, n_physical:]
    retained_action = linalg.solve(krr, krp, assume_a="sym")
    physical_k = kpp-kpr@retained_action
    ep, up = linalg.eigh(physical_k)
    er, ur = linalg.eigh(krr)
    if (ep[0] <= 1e-10*ep[-1] or
            np.min(np.abs(er)) <= 1e-12*np.max(np.abs(er))):
        raise ValueError("positive physical/regular retained split unavailable")

    # C integrates each physical gap against a dimensionless line basis.
    # Divide by a fixed translation scale and line length (1 m); convert
    # rotations into lengths with the declared characteristic length.
    row_scale = np.tile(np.r_[np.ones(3),
                              np.full(3, characteristic_length)],
                        space.count)/displacement_scale
    cbar = (row_scale[:, None]*rows)*scales[None, :]
    cp, cr = cbar[:, :n_physical], cbar[:, n_physical:]
    cl = cp-cr@retained_action
    bp = cl@(up/np.sqrt(ep)[None, :])
    br = cr@(ur/np.sqrt(np.abs(er))[None, :])
    signed = bp@bp.T+(br*np.sign(er)[None, :])@br.T
    direct = cbar@linalg.solve(kbar, cbar.T, assume_a="sym")
    decomposition_error = float(np.linalg.norm(signed-direct)/
                                max(np.linalg.norm(direct), 1e-30))
    if decomposition_error > 1e-9:
        raise ArithmeticError("retained interface decomposition inconsistent")

    scalar_mass = np.zeros((space.count, space.count))
    points, weights = np.polynomial.legendre.leggauss(3)
    for left, right in zip(space.knots[:-1], space.knots[1:]):
        half, center = .5*(right-left), .5*(right+left)
        for point, weight in zip(points, weights):
            phi = space.values(float(center+half*point))
            scalar_mass += half*weight*np.outer(phi, phi)
    mass = np.kron(scalar_mass, np.eye(6))
    em, um = linalg.eigh(mass)
    if em[0] <= 1e-12*em[-1]:
        raise ValueError("multiplier L2 Gram is deficient")
    inv_sqrt = (um/np.sqrt(em)[None, :])@um.T
    bpw, brw = inv_sqrt@bp, inv_sqrt@br
    left, singular, _ = linalg.svd(
        np.column_stack((bpw, brw)), full_matrices=False)
    mode = inv_sqrt@left[:, -1]
    mode = mode.reshape(space.count, 6)
    channel_shares = [float(mode[:, axis]@scalar_mass@mode[:, axis])
                      for axis in range(6)]
    physical_share = float(np.linalg.norm(left[:, -1]@bpw)**2/
                           singular[-1]**2)
    if not -1e-10 <= physical_share <= 1.+1e-10:
        raise ArithmeticError("invalid weakest-mode work decomposition")
    physical_share = float(np.clip(physical_share, 0., 1.))
    weighted = np.column_stack((bpw, brw))
    channel_beta = []
    for axis in range(6):
        selected = np.arange(axis, len(rows), 6)
        channel_beta.append(float(linalg.svdvals(weighted[selected, :])[-1]))
    transverse_rows = np.array([6*i+axis for i in range(space.count)
                                for axis in (2, 3, 4)])
    transverse_left, transverse_singular, _ = linalg.svd(
        weighted[transverse_rows, :], full_matrices=False)
    transverse_mode = transverse_left[:, -1]
    transverse_shares = [float(np.linalg.norm(transverse_mode[axis::3])**2)
                         for axis in range(3)]
    transverse_physical = float(np.linalg.norm(
        transverse_mode@bpw[transverse_rows, :])**2/
        transverse_singular[-1]**2)
    if not -1e-10 <= transverse_physical <= 1.+1e-10:
        raise ArithmeticError("invalid transverse work decomposition")
    transverse_physical = float(np.clip(transverse_physical, 0., 1.))
    dual_diagnostic = None
    if kind in ("piecewise_constant", "continuous_linear", "dual_linear"):
        # The L2 quotient above shrinks for high-frequency dual loads even
        # when a trace pair is stable. Compare the same absolute-energy
        # action with an analytically bounded reference H^(-1/2) dual norm.
        gram_builder={"piecewise_constant":p0_dual_gram,
                      "continuous_linear":p1_dual_gram,
                      "dual_linear":dual_linear_gram}[kind]
        lower,upper=gram_builder(intervals,1.,4095)
        if lower.shape != (space.count,space.count):
            raise ArithmeticError("dual norm and multiplier space disagree")
        transverse_action=np.column_stack((bp[transverse_rows],
                                            br[transverse_rows]))
        transverse_action=transverse_action@transverse_action.T
        transverse_action=(transverse_action+transverse_action.T)/2
        lower3=np.kron(lower,np.eye(3))
        upper3=np.kron(upper,np.eye(3))
        low_values,low_vectors=linalg.eigh(transverse_action,upper3)
        high_values=linalg.eigh(transverse_action,lower3,eigvals_only=True)
        if low_values[0] < -1e-9*max(1.,low_values[-1]):
            raise ArithmeticError("negative absolute-energy dual action")
        # The earlier transverse screen excludes Mz. Test the physically
        # polar-yaw channel separately with the same moment/length scaling
        # and bounded dual Gram; it is not a mixed-channel stability proof.
        yaw_rows=np.arange(5,len(rows),6)
        yaw_bp,yaw_br=bp[yaw_rows],br[yaw_rows]
        yaw_action=yaw_bp@yaw_bp.T+yaw_br@yaw_br.T
        yaw_action=(yaw_action+yaw_action.T)/2
        yaw_low,yaw_vectors=linalg.eigh(yaw_action,upper)
        yaw_high=linalg.eigh(yaw_action,lower,eigvals_only=True)
        if yaw_low[0] < -1e-9*max(1.,yaw_low[-1]):
            raise ArithmeticError("negative yaw absolute-energy dual action")
        yaw_mode=yaw_vectors[:,0]
        yaw_significant=np.abs(yaw_mode)>1e-8*np.max(np.abs(yaw_mode))
        yaw_signs=np.sign(yaw_mode[yaw_significant])
        yaw_work=float(yaw_mode@yaw_action@yaw_mode)
        yaw_physical_share=float(np.linalg.norm(yaw_mode@yaw_bp)**2/
                                 max(yaw_work,1e-300))
        if not -1e-10 <= yaw_physical_share <= 1.+1e-10:
            raise ArithmeticError("invalid yaw work decomposition")
        mx_rows=np.arange(3,len(rows),6)
        mx_bp,mx_br=bp[mx_rows],br[mx_rows]
        mx_action=mx_bp@mx_bp.T+mx_br@mx_br.T
        mx_action=(mx_action+mx_action.T)/2
        mx_low,mx_vectors=linalg.eigh(mx_action,upper)
        mx_high=linalg.eigh(mx_action,lower,eigvals_only=True)
        if mx_low[0] < -1e-9*max(1.,mx_low[-1]):
            raise ArithmeticError("negative Mx absolute-energy dual action")
        mx_mode=mx_vectors[:,0]
        mx_significant=np.abs(mx_mode)>1e-8*np.max(np.abs(mx_mode))
        mx_signs=np.sign(mx_mode[mx_significant])
        mx_work=float(mx_mode@mx_action@mx_mode)
        mx_physical_share=float(np.linalg.norm(mx_mode@mx_bp)**2/
                                 max(mx_work,1e-300))
        if not -1e-10 <= mx_physical_share <= 1.+1e-10:
            raise ArithmeticError("invalid Mx work decomposition")
        # The saved torque-channel ablation identifies Fy/Mx/Mz together as
        # the load-carrying subspace. Its minimum is more relevant to torque
        # than the all-channel or Fz/Mx/My minimum, but remains diagnostic.
        torque_rows=np.array([6*i+axis for i in range(space.count)
                              for axis in (1,3,5)])
        torque_bp,torque_br=bp[torque_rows],br[torque_rows]
        torque_action=torque_bp@torque_bp.T+torque_br@torque_br.T
        torque_action=(torque_action+torque_action.T)/2
        torque_low,torque_vectors=linalg.eigh(torque_action,upper3)
        torque_high=linalg.eigh(torque_action,lower3,eigvals_only=True)
        if torque_low[0] < -1e-9*max(1.,torque_low[-1]):
            raise ArithmeticError("negative torque-subspace dual action")
        torque_mode=torque_vectors[:,0].reshape(space.count,3)
        torque_shares=np.array([float(torque_mode[:,axis]@upper@
                                      torque_mode[:,axis]) for axis in range(3)])
        torque_work=float(torque_vectors[:,0]@torque_action@
                          torque_vectors[:,0])
        torque_physical_share=float(np.linalg.norm(
            torque_vectors[:,0]@torque_bp)**2/max(torque_work,1e-300))
        if not -1e-10 <= torque_physical_share <= 1.+1e-10:
            raise ArithmeticError("invalid torque-subspace work decomposition")
        mode3=low_vectors[:,0].reshape(space.count,3)
        shares=[float(mode3[:,axis]@upper@mode3[:,axis])
                for axis in range(3)]
        fz=mode3[:,0]
        significant=np.abs(fz)>1e-8*np.max(np.abs(fz))
        signs=np.sign(fz[significant])
        dual_diagnostic={
            "norm":"reference Neumann H^(-1/2); development diagnostic",
            "space":kind,
            "last_integrated_cosine_mode":4095,
            "transverse_beta_lower":float(np.sqrt(max(0.,low_values[0]))),
            "transverse_beta_upper":float(np.sqrt(max(0.,high_values[0]))),
            "gram_relative_enclosure":float(
                linalg.eigvalsh(upper3,lower3)[-1]-1.),
            "weakest_fz_norm_share":shares[0]/sum(shares),
            "weakest_fz_sign_changes":int(np.count_nonzero(
                signs[1:] != signs[:-1])),
            "weakest_physical_work_share":float(
                np.linalg.norm(low_vectors[:,0]@bp[transverse_rows])**2/
                max(low_vectors[:,0]@transverse_action@low_vectors[:,0],
                    1e-300)),
            "yaw_mz_beta_lower":float(np.sqrt(max(0.,yaw_low[0]))),
            "yaw_mz_beta_upper":float(np.sqrt(max(0.,yaw_high[0]))),
            "yaw_mz_sign_changes":int(np.count_nonzero(
                yaw_signs[1:] != yaw_signs[:-1])),
            "yaw_mz_physical_work_share":float(np.clip(
                yaw_physical_share,0.,1.)),
            "mx_beta_lower":float(np.sqrt(max(0.,mx_low[0]))),
            "mx_beta_upper":float(np.sqrt(max(0.,mx_high[0]))),
            "mx_sign_changes":int(np.count_nonzero(
                mx_signs[1:] != mx_signs[:-1])),
            "mx_physical_work_share":float(np.clip(mx_physical_share,0.,1.)),
            "torque_fy_mx_mz_beta_lower":float(np.sqrt(max(0.,torque_low[0]))),
            "torque_fy_mx_mz_beta_upper":float(np.sqrt(max(0.,torque_high[0]))),
            "torque_mode_fy_mx_mz_norm_shares":(
                torque_shares/torque_shares.sum()).tolist(),
            "torque_mode_physical_work_share":float(np.clip(
                torque_physical_share,0.,1.)),
        }
        if loaded_mode_overlap:
            shell_size=6*shell.panel.node_count
            tip_rx=shell_size+6*beam.n_half_cells+3
            load=np.zeros(kkt.primal_tangent.shape[0])
            load[tip_rx]=100.
            rhs=np.r_[load[free]*kkt.coordinate_scale[:n]/
                       kkt.energy_scale,np.zeros(len(rows))]
            scaled_solution=linalg.solve(kkt.scaled_system,rhs,
                                         assume_a="sym")
            physical_solution=kkt.coordinate_scale*scaled_solution
            full=np.zeros(len(load))
            full[free]=physical_solution[:n]
            multiplier=physical_solution[n:]
            equilibrium=kkt.primal_tangent@full+(
                kkt.constraint_jacobian.T@multiplier)-load
            gap=kkt.constraint_jacobian@full
            scaled_error=float(np.linalg.norm(equilibrium[free])/100.)
            if scaled_error>1e-8 or np.max(np.abs(gap))>1e-8:
                raise ArithmeticError("loaded torque mode has invalid equilibrium")
            # cbar = diag(row_scale) C diag(scales); the normalized dual
            # coefficient is proportional to lambda/row_scale. The omitted
            # common energy factor cancels from projection and norm shares.
            normalized_multiplier=(multiplier/row_scale)[torque_rows]
            loaded_norm=float(normalized_multiplier@upper3@
                              normalized_multiplier)
            if loaded_norm<=0.:
                raise ArithmeticError("loaded torque multiplier norm vanished")
            overlap=float((torque_vectors[:,0]@upper3@
                           normalized_multiplier)**2/loaded_norm)
            mx_multiplier=(multiplier/row_scale)[mx_rows]
            loaded_mx_norm=float(mx_multiplier@upper@mx_multiplier)
            mx_overlap=float((mx_mode@upper@mx_multiplier)**2/
                             max(loaded_mx_norm,1e-300))
            loaded_channels=normalized_multiplier.reshape(space.count,3)
            loaded_shares=np.array([float(loaded_channels[:,axis]@upper@
                                          loaded_channels[:,axis])
                                    for axis in range(3)])/loaded_norm
            if (not -1e-10<=overlap<=1.+1e-10 or
                    not -1e-10<=mx_overlap<=1.+1e-10):
                raise ArithmeticError("loaded torque mode overlap invalid")
            dual_diagnostic.update({
                "loaded_torque_tip_rx_rad":float(full[tip_rx]),
                "loaded_torque_scaled_equilibrium_error":scaled_error,
                "loaded_torque_weak_mode_dual_overlap_fraction":float(
                    np.clip(overlap,0.,1.)),
                "loaded_torque_mx_weak_mode_dual_overlap_fraction":float(
                    np.clip(mx_overlap,0.,1.)),
                "loaded_torque_fy_mx_mz_dual_norm_shares":
                    loaded_shares.tolist(),
            })
    return {
        "topology": topology, "shell_x_intervals": 2*n_s,
        "shell_y_intervals": n_y,
        "native_half_cells": native_half_cells,
        "shell_thickness_m": shell.panel.thickness,
        "beam_reference_z_m": beam_z_offset,
        "beam_torsion_constant_m4": beam_torsion_constant,
        "line_intervals": intervals, "space": kind,
        "shell_port": shell_port,
        "confined_to_attachment": confined,
        "row_count": len(rows),
        "row_rank": int(np.linalg.matrix_rank(rows)),
        "shell_only_row_rank": int(np.linalg.matrix_rank(
            rows[:,free < 6*shell.panel.node_count])),
        "beam_only_row_rank": int(np.linalg.matrix_rank(
            rows[:,free >= 6*shell.panel.node_count])),
        "physical_stiffness_min_eigen": float(ep[0]),
        "retained_negative_modes": int(np.count_nonzero(er < 0.)),
        "decomposition_relative_error": decomposition_error,
        "characteristic_length_m": characteristic_length,
        "minimum_algorithmic_beta": float(singular[-1]),
        "weakest_mode_channel_l2_shares": channel_shares,
        "weakest_mode_physical_work_share": physical_share,
        "weakest_mode_retained_work_share": 1.-physical_share,
        "single_channel_beta_fx_fy_fz_mx_my_mz": channel_beta,
        "transverse_fz_mx_my_beta": float(transverse_singular[-1]),
        "transverse_mode_fz_mx_my_l2_shares": transverse_shares,
        "transverse_mode_physical_work_share": transverse_physical,
        "transverse_mode_retained_work_share": 1.-transverse_physical,
        "p0_dual_diagnostic":dual_diagnostic,
        "density_dual_diagnostic":dual_diagnostic,
        "scope": "finite native six-channel algorithmic norm; not physical inf-sup acceptance",
    }


def channel_split_reference_modes(
        topology: str, *, coarse_intervals: int = 4,
        fine_intervals: int = 8, fine_channels: tuple[int, ...] = (3,),
        n_s: int = 4, n_y: int = 4, native_half_cells: int = 16,
        characteristic_length: float = .46,
        shell_port: str = "physical_strip_005_005",
        thickness: float = .07, beam_z_offset: float = 0.,
        beam_torsion_constant: float | None = None) -> dict:
    """Absolute-energy dual-norm screen for a bounded mixed-channel P0 pair.

    The H^(-1/2) Gram follows the actual interval count of each physical
    wrench channel. This is a finite-model diagnostic, not inf-sup proof.
    """
    if not np.isfinite(characteristic_length) or characteristic_length <= 0.:
        raise ValueError("positive physical characteristic length required")
    shell=FlatE4TrialPanel(make_shell(n_s,topology,n_y=n_y,
                                      thickness=thickness))
    beam=build(native_half_cells,z_offset=beam_z_offset,
               torsion_constant=beam_torsion_constant)
    kkt=assemble_channel_split_reference_kkt(
        shell,beam,coarse_intervals=coarse_intervals,
        fine_intervals=fine_intervals,fine_channels=fine_channels,
        shell_port=shell_port)
    free=kkt.free_primal
    n=len(free)
    if n+kkt.constraint_jacobian.shape[0] > 1200:
        raise ValueError("bounded dense interface diagnostic exceeds 1200 rows")
    physical_count=int(np.count_nonzero(
        free < 6*shell.panel.node_count+beam.context.nodal_count))
    displacement_scale=1e6/(210e9*.07)
    scales=kkt.coordinate_scale[:n]
    k=kkt.primal_tangent[np.ix_(free,free)]
    kbar=(scales[:,None]*k)*scales[None,:]/kkt.energy_scale
    if np.linalg.norm(kbar-kbar.T)>1e-10*np.linalg.norm(kbar):
        raise ValueError("nonsymmetric reference tangent cannot use energy norm")
    kpp=kbar[:physical_count,:physical_count]
    kpr=kbar[:physical_count,physical_count:]
    krp=kbar[physical_count:,:physical_count]
    krr=kbar[physical_count:,physical_count:]
    retained_action=linalg.solve(krr,krp,assume_a="sym")
    physical_k=kpp-kpr@retained_action
    ep,up=linalg.eigh(physical_k)
    er,ur=linalg.eigh(krr)
    if (ep[0]<=1e-10*ep[-1] or
            np.min(np.abs(er))<=1e-12*np.max(np.abs(er))):
        raise ValueError("positive physical/regular retained split unavailable")
    channels=tuple(j for _ in range(coarse_intervals)
                   for j in range(6) if j not in fine_channels)+tuple(
                   j for _ in range(fine_intervals) for j in fine_channels)
    if len(channels)!=kkt.constraint_jacobian.shape[0]:
        raise AssertionError("split channel row inventory differs")
    row_scale=np.array([1. if j<3 else characteristic_length
                        for j in channels])/displacement_scale
    cbar=(row_scale[:,None]*kkt.constraint_jacobian[:,free])*scales[None,:]
    cp,cr=cbar[:,:physical_count],cbar[:,physical_count:]
    cl=cp-cr@retained_action
    bp=cl@(up/np.sqrt(ep)[None,:])
    br=cr@(ur/np.sqrt(np.abs(er))[None,:])
    signed=bp@bp.T+(br*np.sign(er)[None,:])@br.T
    direct=cbar@linalg.solve(kbar,cbar.T,assume_a="sym")
    decomposition_error=float(np.linalg.norm(signed-direct)/
                              max(np.linalg.norm(direct),1e-30))
    if decomposition_error>1e-9:
        raise ArithmeticError("retained interface decomposition inconsistent")
    lower=np.zeros((len(channels),len(channels)))
    upper=np.zeros_like(lower)
    for channel in range(6):
        indices=np.flatnonzero(np.array(channels)==channel)
        count=fine_intervals if channel in fine_channels else coarse_intervals
        if len(indices)!=count:
            raise AssertionError("per-channel interval inventory differs")
        lo,hi=p0_dual_gram(count,1.,4095)
        lower[np.ix_(indices,indices)]=lo
        upper[np.ix_(indices,indices)]=hi
    selected=np.array([i for i,j in enumerate(channels) if j in (2,3,4)])
    transverse=np.column_stack((bp[selected],br[selected]))
    action=transverse@transverse.T
    action=(action+action.T)/2
    lo=lower[np.ix_(selected,selected)]
    hi=upper[np.ix_(selected,selected)]
    eigen,vectors=linalg.eigh(action,hi)
    bound=linalg.eigh(action,lo,eigvals_only=True)
    if eigen[0]<-1e-9*max(1.,eigen[-1]):
        raise ArithmeticError("negative absolute-energy dual action")
    mode=vectors[:,0]
    selected_channels=np.array(channels)[selected]
    channel_shares={}
    sign_changes={}
    for channel in (2,3,4):
        indices=np.flatnonzero(selected_channels==channel)
        values=mode[indices]
        channel_shares[channel]=float(values@hi[np.ix_(indices,indices)]@values)
        significant=np.abs(values)>1e-8*max(np.max(np.abs(values)),1e-300)
        signs=np.sign(values[significant])
        sign_changes[channel]=int(np.count_nonzero(signs[1:]!=signs[:-1]))
    return {
        "topology":topology,"shell_port":shell_port,
        "shell_x_intervals":2*n_s,
        "shell_y_intervals":n_y,"native_half_cells":native_half_cells,
        "shell_thickness_m":thickness,"beam_reference_z_m":beam_z_offset,
        "beam_torsion_constant_m4":beam_torsion_constant,
        "coarse_line_intervals":coarse_intervals,
        "fine_line_intervals":fine_intervals,
        "fine_channels":fine_channels,"row_count":len(channels),
        "row_rank":int(np.linalg.matrix_rank(kkt.constraint_jacobian[:,free])),
        "shell_only_row_rank":int(np.linalg.matrix_rank(
            kkt.constraint_jacobian[:,free[free<6*shell.panel.node_count]])),
        "beam_only_row_rank":int(np.linalg.matrix_rank(
            kkt.constraint_jacobian[:,free[free>=6*shell.panel.node_count]])),
        "decomposition_relative_error":decomposition_error,
        "transverse_beta_lower":float(np.sqrt(max(0.,eigen[0]))),
        "transverse_beta_upper":float(np.sqrt(max(0.,bound[0]))),
        "gram_relative_enclosure":float(linalg.eigvalsh(hi,lo)[-1]-1.),
        "weakest_channel_dual_norm_shares_fz_mx_my":tuple(
            channel_shares[j]/sum(channel_shares.values()) for j in (2,3,4)),
        "weakest_channel_sign_changes_fz_mx_my":tuple(
            sign_changes[j] for j in (2,3,4)),
        "weakest_physical_work_share":float(
            np.linalg.norm(mode@bp[selected])**2/
            max(mode@action@mode,1e-300)),
        "scope":"finite absolute-energy dual norm including E4 completion; not inf-sup acceptance",
    }


def finite_band_mixed_norm(
        topology: str, *, shell_longitudinal_half_cells: int = 4,
        shell_transverse_intervals: int = 8,
        native_half_cells: int = 16,
        p0_intervals: int = 4, dual_intervals: int = 8,
        thickness: float = .035, beam_z_offset: float = .0675,
        beam_torsion_constant: float | None = None,
        characteristic_length: float = .46,
        last_mode: int = 4095,
        dual_norm_profile: str = "finite_band") -> dict:
    """Finite-band mixed-space norm diagnostic at the virgin reference.

    Width-mean translation/director gaps use an H^1 trace test norm, so
    Fx/Fy/Fz/Mx/My densities use its Neumann H^-1 dual. Gradient-built
    polar yaw is only L2 controlled, so Mz uses a normalized L2 dual.
    This absolute-energy action includes the indefinite native retained
    block. It is not a continuum inf-sup certificate or a space selection.
    """
    if not np.isfinite(characteristic_length) or characteristic_length <= 0.:
        raise ValueError("positive characteristic length required")
    if dual_norm_profile not in ("finite_band", "uniform_hminus_half"):
        raise ValueError("declared finite-band or legacy diagnostic norm required")
    shell = FlatE4TrialPanel(make_shell(
        shell_longitudinal_half_cells, topology,
        n_y=shell_transverse_intervals, thickness=thickness))
    beam = build(native_half_cells, z_offset=beam_z_offset,
                 torsion_constant=beam_torsion_constant)
    p0 = LineMultiplierSpace(
        "piecewise_constant", tuple(np.linspace(0., 1., p0_intervals+1)))
    dual = LineMultiplierSpace(
        "dual_linear", tuple(np.linspace(0., 1., dual_intervals+1)))
    spaces = (p0, dual, p0, dual, p0, dual)
    kkt = assemble_channel_basis_reference_kkt(
        shell, beam, spaces, shell_port=TRANSVERSE_BAND_PORT,
        shell_support_policy="full_left_edge")
    free = kkt.free_primal
    n = len(free)
    shell_size = 6*shell.panel.node_count
    physical_count = int(np.count_nonzero(
        free < shell_size+beam.context.nodal_count))
    coordinate_scale = kkt.coordinate_scale[:n]
    stiffness = kkt.primal_tangent[np.ix_(free, free)]
    scaled_k = ((coordinate_scale[:, None]*stiffness)*
                coordinate_scale[None, :])/kkt.energy_scale
    if np.linalg.norm(scaled_k-scaled_k.T) > 1e-10*np.linalg.norm(scaled_k):
        raise ValueError("nonsymmetric reference tangent lacks energy norm")
    kpp = scaled_k[:physical_count, :physical_count]
    kpr = scaled_k[:physical_count, physical_count:]
    krp = scaled_k[physical_count:, :physical_count]
    krr = scaled_k[physical_count:, physical_count:]
    retained_action = linalg.solve(krr, krp, assume_a="sym")
    physical_k = kpp-kpr@retained_action
    physical_eigen, physical_vectors = linalg.eigh(physical_k)
    retained_eigen, retained_vectors = linalg.eigh(krr)
    if (physical_eigen[0] <= 1e-10*physical_eigen[-1] or
            np.min(np.abs(retained_eigen)) <=
            1e-12*np.max(np.abs(retained_eigen))):
        raise ValueError("positive physical/regular retained split unavailable")
    channels = tuple(axis for axis, space in enumerate(spaces)
                     for _ in range(space.count))
    row_scale = np.array([
        1. if axis < 3 else characteristic_length for axis in channels])
    displacement_scale = 1e6/(210e9*.07)
    scaled_c = ((row_scale[:, None]/displacement_scale)*
                kkt.constraint_jacobian[:, free])*coordinate_scale[None, :]
    cp, cr = scaled_c[:, :physical_count], scaled_c[:, physical_count:]
    condensed_c = cp-cr@retained_action
    bp = condensed_c@(physical_vectors/
                      np.sqrt(physical_eigen)[None, :])
    br = cr@(retained_vectors/
             np.sqrt(np.abs(retained_eigen))[None, :])
    action = bp@bp.T+br@br.T
    action = (action+action.T)/2
    signed = bp@bp.T+(br*np.sign(retained_eigen)[None, :])@br.T
    direct = scaled_c@linalg.solve(scaled_k, scaled_c.T, assume_a="sym")
    decomposition_error = float(np.linalg.norm(signed-direct)/
                                max(np.linalg.norm(direct), 1e-30))
    if decomposition_error > 1e-9:
        raise ArithmeticError("physical/retained work decomposition differs")
    order = 1. if dual_norm_profile == "finite_band" else 0.5
    p0_lower, p0_upper = p0_dual_gram(
        p0_intervals, 1., last_mode, sobolev_order=order)
    dual_lower, dual_upper = dual_linear_gram(
        dual_intervals, 1., last_mode, sobolev_order=order)
    yaw_lower, yaw_upper = ((dual_linear_l2_gram(dual_intervals, 1.),)*2
                            if dual_norm_profile == "finite_band" else
                            (dual_lower, dual_upper))
    lower = np.zeros((len(channels), len(channels)))
    upper = np.zeros_like(lower)
    channel_slices = []
    start = 0
    for axis, space in enumerate(spaces):
        stop = start+space.count
        lo, hi = ((p0_lower, p0_upper) if space is p0 else
                  (yaw_lower, yaw_upper) if axis == 5 else
                  (dual_lower, dual_upper))
        lower[start:stop, start:stop] = lo
        upper[start:stop, start:stop] = hi
        channel_slices.append(slice(start, stop))
        start = stop
    eigen, vectors = linalg.eigh(action, upper)
    upper_eigen = linalg.eigh(action, lower, eigvals_only=True)
    if eigen[0] < -1e-9*max(1., eigen[-1]):
        raise ArithmeticError("negative absolute-energy dual action")
    mode = vectors[:, 0]
    channel_shares = []
    channel_sign_changes = []
    for part in channel_slices:
        entries = mode[part]
        channel_shares.append(float(
            entries@upper[part, part]@entries))
        significant = np.abs(entries) > 1e-8*max(
            np.max(np.abs(entries)), 1e-300)
        signs = np.sign(entries[significant])
        channel_sign_changes.append(int(np.count_nonzero(
            signs[1:] != signs[:-1])))
    physical_share = float(np.linalg.norm(mode@bp)**2/
                           max(mode@action@mode, 1e-300))
    def restricted_beta(axes: tuple[int, ...]) -> tuple[float, float]:
        selected = np.array([index for axis in axes
                             for index in range(
                                 channel_slices[axis].start,
                                 channel_slices[axis].stop)])
        sub = np.ix_(selected, selected)
        lo = linalg.eigh(action[sub], upper[sub], eigvals_only=True)[0]
        hi = linalg.eigh(action[sub], lower[sub], eigvals_only=True)[0]
        return float(np.sqrt(max(0., lo))), float(np.sqrt(max(0., hi)))
    torque_beta = restricted_beta((1, 3, 5))
    shear_beta = restricted_beta((2, 4))
    def scaled_rank(columns: np.ndarray) -> int:
        values = np.linalg.svd(scaled_c[:, columns], compute_uv=False)
        return int(np.count_nonzero(values > 1e-10*values[0]))
    return {
        "topology": topology,
        "shell_x_elements": 2*shell_longitudinal_half_cells,
        "shell_y_intervals": shell_transverse_intervals,
        "native_half_cells": native_half_cells,
        "p0_intervals": p0_intervals,
        "dual_intervals": dual_intervals,
        "row_count": len(channels),
        "row_rank": scaled_rank(np.ones(n, dtype=bool)),
        "shell_only_row_rank": scaled_rank(free < shell_size),
        "beam_only_row_rank": scaled_rank(free >= shell_size),
        "row_rank_threshold": "relative scaled singular value 1e-10",
        "dual_norm_profile": dual_norm_profile,
        "dual_norm_by_channel": (("H^-1",)*5+("L2",)
                                 if dual_norm_profile == "finite_band" else
                                 ("H^-1/2",)*6),
        "characteristic_length_m": characteristic_length,
        "decomposition_relative_error": decomposition_error,
        "minimum_beta_lower": float(np.sqrt(max(0., eigen[0]))),
        "minimum_beta_upper": float(np.sqrt(max(0., upper_eigen[0]))),
        "torque_channels_fy_mx_mz_beta_lower_upper": torque_beta,
        "shear_channels_fz_my_beta_lower_upper": shear_beta,
        "weakest_channel_norm_shares_fx_fy_fz_mx_my_mz": tuple(
            channel_shares),
        "weakest_channel_sign_changes_fx_fy_fz_mx_my_mz": tuple(
            channel_sign_changes),
        "weakest_physical_work_share": physical_share,
        "weakest_retained_work_share": 1.-physical_share,
        "scope": "finite mixed algorithmic absolute-energy norm; no inf-sup or physical accuracy acceptance",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-translation", action="store_true")
    parser.add_argument("--axial-wave", action="store_true")
    parser.add_argument("--include-dual", action="store_true")
    parser.add_argument("--paired-refinement", action="store_true")
    parser.add_argument("--paired-shell-intervals", type=int,
                        choices=(1, 2, 4, 8), nargs="+")
    parser.add_argument("--shell-transverse-intervals", type=int,
                        choices=(2, 4, 8), default=2)
    parser.add_argument("--kind", choices=("piecewise_constant",
                                           "continuous_linear", "dual_linear"))
    parser.add_argument("--shell-support-policy",
                        choices=("legacy_three_left_nodes", "full_left_edge",
                                 "both_xy_edges"),
                        default="legacy_three_left_nodes")
    parser.add_argument("--sinusoidal-axial-density", action="store_true")
    parser.add_argument("--transverse-force-z", action="store_true")
    arguments = parser.parse_args()
    if sum((arguments.axial_wave, arguments.sinusoidal_axial_density,
            arguments.transverse_force_z)) > 1:
        parser.error("select only one diagnostic load")
    if arguments.paired_shell_intervals and not arguments.paired_refinement:
        parser.error("paired shell intervals require paired refinement")
    shell_port = ("local_translation" if arguments.local_translation
                  else "rail")
    load_kind = ("transverse_force_z" if arguments.transverse_force_z else
                 "sinusoidal_axial_density" if arguments.sinusoidal_axial_density
                 else "axial_wave" if arguments.axial_wave else "torsion")
    kinds = ("piecewise_constant", "continuous_linear")
    if arguments.include_dual:
        kinds += ("dual_linear",)
    if arguments.kind is not None:
        kinds = (arguments.kind,)
    paired_sizes = arguments.paired_shell_intervals or (1, 2, 4)
    grids = ((n_s, 4*n_s, n_s) for n_s in paired_sizes) if (
        arguments.paired_refinement) else ((4, 16, intervals)
                                          for intervals in (2, 4, 8))
    grids = tuple(grids)
    rows = [one_case(topology, intervals, kind, shell_port=shell_port,
                     load_kind=load_kind, n_s=n_s,
                     native_half_cells=native_half_cells,
                     n_y=arguments.shell_transverse_intervals,
                     shell_support_policy=arguments.shell_support_policy)
            for topology in ("Q4", "S3-V2D")
            for kind in kinds
            for n_s, native_half_cells, intervals in grids]
    print(json.dumps({"status": "UNREGISTERED_DIAGNOSTIC",
                      "screen": "paired_refinement" if arguments.paired_refinement
                      else "fixed_primal",
                      "shell_transverse_intervals": arguments.shell_transverse_intervals,
                      "shell_support_policy": arguments.shell_support_policy,
                      "shell_port": shell_port, "load_kind": load_kind,
                      "cases": rows},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
