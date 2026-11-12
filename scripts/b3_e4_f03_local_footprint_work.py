"""Bounded G0 comparison of rigid and distributing 0.1 m footprints.

The same E4 panel and retained B3 beam supply both local tangent maps.
This is a point-station mechanics diagnostic, not a solved full line bond.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import spsolve
from threadpoolctl import threadpool_info, threadpool_limits

from anyfesolver._ge_beam3_mixed_ad import Jet2, matvec, so3_exp, transpose
from anyfesolver._ge_beam3_pose_joint import RigidPoseJoint
from scripts._b3_e4_f03_coupled_point_work_candidate import directional_point_work
from scripts._b3_e4_f03_e4_trial_panel import FlatE4TrialPanel
from scripts._b3_e4_f03_native_fixture import SECTION_DIAGONAL, build
from scripts._b3_e4_f03_native_spatial_port import (
    NativeHalfCellSpatialPort, NativeNodalSpatialPort,
)
from scripts._b3_e4_f03_shell_finite_pose_candidate import reference_nodes
from scripts._b3_e4_f03_shell_spatial_linear_port import physical_shell_stiffness
from scripts._b3_e4_f03_shell_strip_pose_candidate import (
    directional_pose_from_e4_total_strip, pose_from_e4_total_strip,
    reference_spatial_trace_strip, strip_weights, _clip, _quadrature,
)
from scripts._b3_e4_f03_shell_trace import make_shell
from scripts.b3_e4_f03_plate_patch_work import (
    c1_mixed_s3_cells, local_grid, raw_e4_mixed_c1_tangent, raw_e4_tangent,
)
from scripts.b3_e4_f03_square_section_control import square_torsion_constant


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array(((0., -z, y), (z, 0., -x), (-y, x, 0.)))


def structured_q4_patch_trace(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Exact Q4 area-mean/director/polar-yaw row on the terminal patch."""
    xs=np.asarray(xs,dtype=float)
    ys=np.asarray(ys,dtype=float)
    if (xs.ndim!=1 or ys.ndim!=1 or len(xs)<2 or len(ys)<2 or
            not np.isfinite(xs).all() or not np.isfinite(ys).all() or
            not np.all(np.diff(xs)>0) or not np.all(np.diff(ys)>0) or
            xs[0]>-.5 or xs[-1]<1.5 or ys[0]>-.5 or ys[-1]<.5 or
            6*len(xs)*len(ys)>2300):
        raise ValueError("bounded structured Q4 plate grid required")
    count=len(xs)*len(ys)
    mean=np.zeros(count)
    gradient=np.zeros((2,count))
    covered=0.
    for j in range(len(ys)-1):
        y0,y1=ys[j:j+2]
        v0,v1=max(float(y0),-.05),min(float(y1),.05)
        if v1<=v0:
            continue
        for i in range(len(xs)-1):
            x0,x1=xs[i:i+2]
            u0,u1=max(float(x0),.9),min(float(x1),1.)
            if u1<=u0:
                continue
            area=(u1-u0)*(v1-v0)
            xi=((u0+u1)/2-x0)/(x1-x0)
            eta=((v0+v1)/2-y0)/(y1-y0)
            values=((1-xi)*(1-eta),xi*(1-eta),xi*eta,(1-xi)*eta)
            dx=(-(1-eta),1-eta,eta,-eta)
            dy=(-(1-xi),-xi,xi,1-xi)
            nodes=(j*len(xs)+i,j*len(xs)+i+1,
                   (j+1)*len(xs)+i+1,(j+1)*len(xs)+i)
            for node,value,gx,gy in zip(nodes,values,dx,dy):
                mean[node]+=area*value
                gradient[0,node]+=area*gx/(x1-x0)
                gradient[1,node]+=area*gy/(y1-y0)
            covered+=area
    if abs(covered-.01)>1e-12:
        raise ArithmeticError("terminal patch is not covered by Q4 grid")
    mean/=.01
    gradient/=.01
    locations=np.array([(x,y,0.) for y in ys for x in xs])
    if (abs(mean.sum()-1.)>1e-11 or
            np.linalg.norm(gradient@np.ones(count))>1e-11 or
            np.linalg.norm(gradient@locations[:,:2]-np.eye(2))>1e-11):
        raise ArithmeticError("terminal patch partition or gradient failed")
    trace=np.zeros((6,6*count))
    for axis in range(3):
        trace[axis,axis::6]=mean
    trace[3,3::6]=mean
    trace[4,4::6]=mean
    trace[5,0::6]=-.5*gradient[1]
    trace[5,1::6]=.5*gradient[0]
    return trace


def structured_q4_point_shape(xs: np.ndarray, ys: np.ndarray,
                              x: float, y: float) -> tuple[np.ndarray,np.ndarray]:
    """Physical bilinear host row, including a point on an element edge."""
    i=int(np.clip(np.searchsorted(xs,x,side="left")-1,0,len(xs)-2))
    j=int(np.clip(np.searchsorted(ys,y,side="left")-1,0,len(ys)-2))
    xi=(x-xs[i])/(xs[i+1]-xs[i])
    eta=(y-ys[j])/(ys[j+1]-ys[j])
    if not (-1e-12<=xi<=1+1e-12 and -1e-12<=eta<=1+1e-12):
        raise ValueError("point is outside structured Q4 plate")
    nodes=np.array((j*len(xs)+i,j*len(xs)+i+1,
                    (j+1)*len(xs)+i+1,(j+1)*len(xs)+i))
    values=np.array(((1-xi)*(1-eta),xi*(1-eta),xi*eta,(1-xi)*eta))
    return nodes,values


def structured_c1_mixed_point_shape(
        xs: np.ndarray, ys: np.ndarray, x: float, y: float,
        s3_cells: frozenset[tuple[int, int]], *,
        diagonal: str = "bl_tr",
        ) -> tuple[np.ndarray, np.ndarray]:
    """Q4 or declared-diagonal S3-V2D host on the bounded C1 grid."""
    if diagonal not in ("bl_tr", "br_tl"):
        raise ValueError("bounded C1 S3 diagonal required")
    nodes, values = structured_q4_point_shape(xs, ys, x, y)
    i = int(np.clip(np.searchsorted(xs, x, side="left")-1, 0, len(xs)-2))
    j = int(np.clip(np.searchsorted(ys, y, side="left")-1, 0, len(ys)-2))
    if (i,j) not in s3_cells:
        return nodes, values
    xi = (x-xs[i])/(xs[i+1]-xs[i])
    eta = (y-ys[j])/(ys[j+1]-ys[j])
    if diagonal == "bl_tr":
        if eta <= xi:
            return nodes[[0,1,2]], np.array((1.-xi, xi-eta, eta))
        return nodes[[0,2,3]], np.array((1.-eta, xi, eta-xi))
    if xi+eta <= 1.:
        return nodes[[0,1,3]], np.array((1.-xi-eta, xi, eta))
    return nodes[[1,2,3]], np.array((1.-eta, xi+eta-1., 1.-xi))


def bond_x_segments(lo_x: float, hi_x: float,
                    half_cells: int | np.ndarray
                    ) -> tuple[tuple[float, float], ...]:
    """Split a shell-host interval at every retained B3 half-cell boundary."""
    if not np.isfinite((lo_x, hi_x)).all() or hi_x <= lo_x:
        raise ValueError("valid shell interval and native half-cell count required")
    if type(half_cells) is int and half_cells > 0:
        abscissae = np.arange(half_cells+1, dtype=float)/half_cells
    else:
        abscissae = np.asarray(half_cells, dtype=float)
        if (abscissae.ndim != 1 or len(abscissae) < 2 or
                not np.isfinite(abscissae).all() or
                abscissae[0] != 0. or abscissae[-1] != 1. or
                not np.all(np.diff(abscissae) > 0.)):
            raise ValueError("valid ordered native half-cell stations required")
    cuts = [float(lo_x)]
    cuts.extend(float(x) for x in abscissae[1:-1]
                if lo_x+1e-14 < x < hi_x-1e-14)
    cuts.append(float(hi_x))
    return tuple(zip(cuts[:-1], cuts[1:]))


def distributed_surface_patch_trace(
        xs: np.ndarray, ys: np.ndarray, beam, *,
        plate_thickness: float,
        x_interval: tuple[float, float] = (.9, 1.)) -> tuple[np.ndarray, dict]:
    """Six exact weak position moments on a bounded physical top-surface bond.

    This linearized G0 candidate ties the E4 top-surface *positions* to the
    native retained B3 bottom-surface positions over one specified material
    interval and y=[-.05,.05]. The terminal interval is the default.
    It does not equate the shell director to the surface normal or beam frame.
    The reference second moment normalizes rotation-like rows so their duals
    are physical moments. No penalty, drilling stiffness or tip-only port is
    introduced.
    """
    if plate_thickness not in (.035,.07):
        raise ValueError("bounded E4 plate thickness required")
    if (not isinstance(x_interval, tuple) or len(x_interval)!=2 or
            not np.isfinite(x_interval).all() or
            not 0.<=x_interval[0]<x_interval[1]<=1.):
        raise ValueError("bounded native material interval required")
    line_lo,line_hi=map(float,x_interval)
    line_center=(line_lo+line_hi)/2
    reference_area=(line_hi-line_lo)*.1
    xs,ys=np.asarray(xs,dtype=float),np.asarray(ys,dtype=float)
    if (xs[0]>line_lo or xs[-1]<line_hi or
            ys[0]>-.05 or ys[-1]<.05):
        raise ValueError("shell grid must cover the complete material strip")
    shell_size=6*len(xs)*len(ys)
    total=shell_size+beam.context.count
    mean=np.zeros((3,total))
    first_moment=np.zeros((3,total))
    inertia=np.zeros((3,3))
    shell_top=plate_thickness/2
    beam_axis_z=shell_top+.05
    base=beam.context.initial.mechanical
    roots,gauss_weights=np.polynomial.legendre.leggauss(3)
    area=0.
    max_reference_gap=0.
    for x0,x1 in zip(xs[:-1],xs[1:]):
        lo_x,hi_x=max(float(x0),line_lo),min(float(x1),line_hi)
        if hi_x<=lo_x:
            continue
        for sx0,sx1 in bond_x_segments(lo_x,hi_x,beam.abscissae):
            for y0,y1 in zip(ys[:-1],ys[1:]):
                lo_y,hi_y=max(float(y0),-.05),min(float(y1),.05)
                if hi_y<=lo_y:
                    continue
                jac=(sx1-sx0)*(hi_y-lo_y)/4.
                for xi,wx in zip(roots,gauss_weights):
                    x=(sx0+sx1)/2+float(xi)*(sx1-sx0)/2
                    element,cell,weight,_=beam._locate(x)
                    port=NativeHalfCellSpatialPort(
                        beam.context,element,cell,weight)
                    high,low,frame=port.pose_pair(base)
                    center=high+low
                    if (np.linalg.norm(center-(x,0.,beam_axis_z))>1e-11 or
                            np.linalg.norm(frame-np.eye(3))>1e-11):
                        raise ArithmeticError("native beam trace reference changed")
                    tangent=port.tangent(base)
                    for eta,wy in zip(roots,gauss_weights):
                        y=(lo_y+hi_y)/2+float(eta)*(hi_y-lo_y)/2
                        shell_indices,shape=structured_q4_point_shape(xs,ys,x,y)
                        surface_offset=np.array((0.,y,-.05))
                        shell_reference=np.array((x,y,shell_top))
                        beam_reference=center+frame@surface_offset
                        max_reference_gap=max(max_reference_gap,float(
                            np.linalg.norm(shell_reference-beam_reference)))
                        if max_reference_gap>1e-11:
                            raise ArithmeticError("solid-like surface reference is open")
                        dg=np.zeros((3,total))
                        for node,value in zip(shell_indices,shape):
                            start=6*int(node)
                            dg[:,start:start+3]+=value*np.eye(3)
                            dg[:,start+3:start+6]+=(
                                -value*shell_top*skew(np.array((0.,0.,1.))))
                        dg[:,shell_size:]-=(np.column_stack((
                            np.eye(3),-skew(surface_offset)))@tangent)
                        arm=np.array((x-line_center,y,0.))
                        quadrature=float(wx*wy*jac)/reference_area
                        mean+=quadrature*dg
                        first_moment+=quadrature*skew(arm)@dg
                        inertia+=quadrature*((arm@arm)*np.eye(3)-
                                             np.outer(arm,arm))
                        area+=float(wx*wy*jac)
    if abs(area-reference_area)>1e-12:
        raise ArithmeticError("distributed bond area changed")
    eigenvalues=np.linalg.eigvalsh(inertia)
    if eigenvalues[0]<=1e-8:
        raise ArithmeticError("degenerate physical footprint inertia")
    constraint=np.vstack((mean,np.linalg.solve(inertia,first_moment)))
    if (not np.isfinite(constraint).all() or
            np.max(np.abs(constraint[:,:shell_size][:,5::6]))>0.):
        raise ArithmeticError("distributed surface row is invalid or uses drilling")
    return constraint,{"reference_gap_m":max_reference_gap,
                       "area_m2":area,
                       "x_interval_m":[line_lo,line_hi],
                       "inertia_eigenvalues_m2":eigenvalues.tolist(),
                       "shell_drilling_column_max":0.,
                       "definition":"mean top-surface position gap and normalized first moment of that gap against native bottom-surface half-cell traces"}


def distributed_surface_patch_trace_panel(
        shell: FlatE4TrialPanel, beam, *,
        x_interval: tuple[float, float] = (.9, 1.)
        ) -> tuple[np.ndarray, dict]:
    """Independent Q4/S3 host-clipped reference oracle for the same patch law.

    Shell hosts, S3 diagonals, native half-cells and physical patch boundaries
    are all split before integrating the surface-position gap and its first
    moment. This is a bounded reference linearization, not an accepted state.
    """
    if not isinstance(shell, FlatE4TrialPanel):
        raise TypeError("bounded actual E4 trial panel required")
    shell.guard()
    beam.context.guard()
    panel = shell.panel
    if (panel.topology not in ("Q4", "S3-V2D") or
            panel.thickness not in (.035, .07) or
            not isinstance(x_interval, tuple) or len(x_interval) != 2 or
            not np.isfinite(x_interval).all() or
            not 0. <= x_interval[0] < x_interval[1] <= 1.):
        raise ValueError("bounded physical Q4/S3 material patch required")
    left, right = map(float, x_interval)
    center_x = (left+right)/2
    expected_area = (right-left)*.1
    nodes = reference_nodes(panel)
    shell_size = 6*panel.node_count
    total = shell_size+beam.context.count
    mean = np.zeros((3, total))
    first_moment = np.zeros((3, total))
    inertia = np.zeros((3, 3))
    area = 0.
    max_reference_gap = 0.
    x_grid = np.linspace(-.5, 1.5, panel.x_interval_count+1)
    y_grid = np.linspace(-.5, .5, panel.n_y+1)
    base = beam.context.initial.mechanical
    for j in range(panel.n_y):
        y0, y1 = y_grid[j:j+2]
        if y1 <= -.05 or y0 >= .05:
            continue
        for i in range(panel.x_interval_count):
            x0, x1 = x_grid[i:i+2]
            if x1 <= left or x0 >= right:
                continue
            a = np.array((x0, y0))
            b = np.array((x1, y0))
            c = np.array((x1, y1))
            d = np.array((x0, y1))
            hosts = ([a, b, c, d],) if panel.topology == "Q4" else (
                [a, b, c], [a, c, d])
            for host in hosts:
                for half_cell in range(beam.n_half_cells):
                    bx0 = half_cell/beam.n_half_cells
                    bx1 = (half_cell+1)/beam.n_half_cells
                    if bx1 <= max(left, x0) or bx0 >= min(right, x1):
                        continue
                    polygon = host
                    for axis, bound, sign in (
                            (0, max(left, bx0), 1),
                            (0, min(right, bx1), -1),
                            (1, -.05, 1), (1, .05, -1)):
                        polygon = _clip(polygon, axis, bound, sign)
                    for weight, point in _quadrature(polygon):
                        x, y = map(float, point)
                        element, cell, fraction, _ = beam._locate(x)
                        port = NativeHalfCellSpatialPort(
                            beam.context, element, cell, fraction)
                        high, low, frame = port.pose_pair(base)
                        shell_indices, shape = panel._shape(x, y)
                        top = panel.thickness/2
                        shell_position = shape@nodes[shell_indices]+(
                            np.array((0., 0., top)))
                        offset = np.array((0., y, -.05))
                        beam_position = high+low+frame@offset
                        max_reference_gap = max(max_reference_gap, float(
                            np.linalg.norm(shell_position-beam_position)))
                        dg = np.zeros((3, total))
                        for node, value in zip(shell_indices, shape):
                            start = 6*int(node)
                            dg[:, start:start+3] += value*np.eye(3)
                            dg[:, start+3:start+6] -= (
                                value*top*skew(np.array((0., 0., 1.))))
                        dg[:, shell_size:] -= (
                            np.column_stack((np.eye(3), -skew(offset))) @
                            port.tangent(base))
                        arm = np.array((x-center_x, y, 0.))
                        mean += weight*dg
                        first_moment += weight*skew(arm)@dg
                        inertia += weight*((arm@arm)*np.eye(3)-
                                           np.outer(arm, arm))
                        area += weight
    if abs(area-expected_area) > 1e-12 or max_reference_gap > 1e-11:
        raise ArithmeticError("bounded surface patch reference changed")
    inertia /= area
    eigenvalues = np.linalg.eigvalsh(inertia)
    if eigenvalues[0] <= 1e-8:
        raise ArithmeticError("degenerate physical footprint inertia")
    rows = np.vstack((mean/area, np.linalg.solve(inertia,
                                                  first_moment/area)))
    if (not np.isfinite(rows).all() or
            np.max(np.abs(rows[:, 5:shell_size:6])) > 1e-14):
        raise ArithmeticError("surface patch row is invalid or uses drilling")
    return rows, {
        "topology": panel.topology,
        "reference_gap_m": max_reference_gap,
        "area_m2": float(area),
        "inertia_eigenvalues_m2": eigenvalues.tolist(),
        "shell_drilling_column_max": float(
            np.max(np.abs(rows[:, 5:shell_size:6]))),
        "definition": "host-clipped Q4/S3 reference oracle for six surface-position moments",
    }


def loaded_surface_patch_panel(topology: str, shell_intervals: int,
                               *, transverse_intervals: int = 4) -> dict:
    """Bounded virgin Q4/S3 and native B3 torque/shear KKT discriminator.

    The beam, 35 mm panel, supports, patch and separate tip loads are fixed.
    The actual E4 trial tangent and retained B3 tangent remain in the KKT;
    this is an exploratory linearization, not a production or G0 solve.
    """
    if (topology not in ("Q4", "S3-V2D") or
            shell_intervals not in (4, 8) or
            transverse_intervals not in (4, 8)):
        raise ValueError("bounded Q4/S3 panel resolution required")
    if shell_intervals == 8 and transverse_intervals == 8:
        raise ValueError("bounded finite-patch KKT row cap exceeded")
    shell = FlatE4TrialPanel(make_shell(
        shell_intervals, topology, n_y=transverse_intervals,
        thickness=.035))
    beam = build(8, z_offset=.0675,
                 torsion_constant=square_torsion_constant())
    shell_size = 6*shell.panel.node_count
    trial = shell.evaluate(np.zeros(shell_size), .95)
    constraint, reference = distributed_surface_patch_trace_panel(shell, beam)
    total = shell_size+beam.context.count
    fixed_shell = {6*j*(shell.panel.x_interval_count+1)+axis
                   for j in range(shell.panel.n_y+1) for axis in range(6)}
    fixed_beam = {shell_size+int(index) for index in beam.context.fixed}
    free = np.array([i for i in range(total)
                     if i not in fixed_shell and i not in fixed_beam])
    if np.linalg.matrix_rank(constraint[:, free]) != 6:
        raise ArithmeticError("supported finite-patch rows lost rank")
    primal = sparse.block_diag(
        (sparse.csr_matrix(trial.tangent), beam.tangent), format="csr")
    physical_shell = physical_shell_stiffness(shell.panel)
    base_residual = np.r_[trial.residual, beam.residual]
    rows = sparse.csr_matrix(constraint[:, free])
    kkt = sparse.bmat([[primal[free, :][:, free], rows.T],
                       [rows, None]], format="csr")
    if kkt.shape[0] > 1000:
        raise ValueError("bounded finite-patch KKT row cap exceeded")
    force_scale = 1e6
    displacement_scale = force_scale/(210e9*.035)
    nodal_scale = np.tile(np.full(6, displacement_scale),
                          shell.panel.node_count+
                          beam.context.nodal_count//6)
    resultants = np.r_[
        np.tile(SECTION_DIAGONAL[:3]*displacement_scale, 2),
        np.tile(SECTION_DIAGONAL[3:]*displacement_scale, 4)]
    retained_scale = np.tile(
        np.r_[np.full(6, displacement_scale), resultants],
        beam.n_half_cells//2)
    primal_scale = np.r_[nodal_scale, retained_scale]
    if len(primal_scale) != total:
        raise ArithmeticError("native retained scaling layout changed")
    scales = np.r_[primal_scale[free], np.full(6, force_scale)]
    energy_scale = force_scale*displacement_scale
    scaled = (sparse.diags(scales)@kkt@sparse.diags(scales)/energy_scale)
    results = {}
    for label, axis, magnitude in (("tip_mx_100nm", 3, 100.),
                                    ("tip_fz_1000n", 2, 1000.)):
        load = np.zeros(total)
        load[shell_size+6*beam.n_half_cells+axis] = magnitude
        effective_load = load-base_residual
        rhs = np.r_[effective_load[free]*scales[:len(free)]/energy_scale,
                    np.zeros(6)]
        answer = spsolve(scaled, rhs)
        state = np.zeros(total)
        state[free] = scales[:len(free)]*answer[:len(free)]
        multipliers = scales[len(free):]*answer[len(free):]
        residual = (base_residual+primal@state+
                    constraint.T@multipliers-load)
        gap = constraint@state
        support = residual[:shell_size].reshape(-1, 6)
        beam_support = residual[shell_size:shell_size+
                                beam.context.nodal_count].reshape(-1, 6)
        shell_nodes = reference_nodes(shell.panel)
        beam_nodes = np.array([(i/beam.n_half_cells, 0., .0675)
                               for i in range(beam.n_half_cells+1)])
        plate_force = support[:, :3].sum(axis=0)
        beam_root_force = beam_support[:, :3].sum(axis=0)
        plate_moment = np.sum(
            np.cross(shell_nodes, support[:, :3])+support[:, 3:], axis=0)
        beam_root_moment = np.sum(
            np.cross(beam_nodes, beam_support[:, :3])+
            beam_support[:, 3:], axis=0)
        tip_force = np.zeros(3)
        tip_moment = np.zeros(3)
        if axis < 3:
            tip_force[axis] = magnitude
            tip_moment = np.cross(beam_nodes[-1], tip_force)
        else:
            tip_moment[axis-3] = magnitude
        force_balance = np.linalg.norm(
            plate_force+beam_root_force+tip_force)
        moment_balance = np.linalg.norm(
            plate_moment+beam_root_moment+tip_moment)
        work_error = abs(state@(primal@state)-effective_load@state)
        physical_shell_work = float(
            state[:shell_size]@physical_shell@state[:shell_size])
        relative_equilibrium = (np.linalg.norm(residual[free])/
                                max(1., np.linalg.norm(load[free])))
        if (not np.isfinite(state).all() or physical_shell_work <= 0. or
                max(relative_equilibrium, np.linalg.norm(gap),
                    work_error, force_balance, moment_balance) > 1e-7):
            raise ArithmeticError("bounded loaded finite-patch check failed")
        results[label] = {
            "tip_work_coordinate": float(state[
                shell_size+6*beam.n_half_cells+axis]),
            "load_work_j": float(load@state),
            "physical_shell_work_j": physical_shell_work,
            "plate_support_force_n": plate_force.tolist(),
            "beam_root_support_force_n": beam_root_force.tolist(),
            "plate_support_moment_nm": plate_moment.tolist(),
            "beam_root_support_moment_nm": beam_root_moment.tolist(),
            "global_force_balance_n": float(force_balance),
            "global_moment_balance_nm": float(moment_balance),
            "scaled_equilibrium_residual": float(relative_equilibrium),
            "weak_gap_norm": float(np.linalg.norm(gap)),
            "work_error_j": float(work_error),
        }
    return {"topology": topology, "shell_intervals": shell_intervals,
            "transverse_intervals": transverse_intervals,
            "beam_half_cells": beam.n_half_cells,
            "kkt_rows": int(kkt.shape[0]), "kkt_nnz": int(kkt.nnz),
            "initial_shell_residual_n": float(np.linalg.norm(trial.residual)),
            "reference": reference, "loads": results,
            "scope": "bounded reference linearization; no G0 or external acceptance"}


def finite_rigid_work(shell: FlatE4TrialPanel, beam, port,
                      points: tuple[tuple[float, float, float], ...],
                      multipliers: np.ndarray) -> dict:
    """Check actual E4 interpolation/native nodal Exp-chart footprint work.

    Only the 12 corner-position rows of the declared rigid law are used.
    Shell director coordinates do not enter that law. The beam frame is the
    native nodal spatial frame, advanced from the same fixed predecessor.
    """
    base = beam.context.initial.mechanical
    shell_size = 6 * shell.panel.node_count
    q = np.zeros(shell_size)
    nodes = reference_nodes(shell.panel)
    nodal = q.reshape(-1, 6)
    nodal[:, 2] = .001 * (nodes[:, 0] + .5) * (1. + nodes[:, 1])
    nodal[:, 3] = .002 * (nodes[:, 0] + .5)
    nodal[:, 4] = -.003 * (nodes[:, 0] + .5)
    baseline = np.zeros(beam.context.count)
    start = 6 * port.node_index
    baseline[start:start+3] = (.0008, -.0005, .0004)
    baseline[start+3:start+6] = (.018, -.012, .009)
    state = beam.context.advance(base, baseline)
    beam_ref_high, beam_ref_low, beam_ref_frame = port.pose_pair(base)
    beam_ref = beam_ref_high + beam_ref_low
    offsets = [beam_ref_frame.T @ (np.asarray(point) - beam_ref)
               for point in points]
    shape_rows = [shell.panel._shape(point[0], point[1]) for point in points]

    u_shell = np.zeros_like(q)
    v_shell = np.zeros_like(q)
    u_shell.reshape(-1, 6)[:, 2] = .007 * (nodes[:, 0] + .5) * nodes[:, 1]
    v_shell.reshape(-1, 6)[:, 0] = .004 * (nodes[:, 0] + .5)
    u_beam = np.zeros(beam.context.count)
    v_beam = np.zeros(beam.context.count)
    u_beam[start:start+6] = (.003, -.002, .004, .07, -.05, .04)
    v_beam[start:start+6] = (-.002, .004, .001, -.06, .03, .05)

    def independent(a: float, b: float, *, motion=None) -> tuple[float, np.ndarray]:
        total = (q + a*u_shell + b*v_shell).reshape(-1, 6)
        changed = beam.context.advance(state, a*u_beam + b*v_beam)
        high, low, frame = port.pose_pair(changed)
        center = high + low
        gaps = []
        for (indices, weights), offset in zip(shape_rows, offsets):
            position = weights @ (nodes[indices] + total[indices, :3])
            if motion is not None:
                rotation, translation = motion
                position = rotation @ position + translation
                transformed_center = rotation @ center + translation
                transformed_frame = rotation @ frame
            else:
                transformed_center, transformed_frame = center, frame
            gaps.extend(transformed_frame.T @ (position-transformed_center) - offset)
        vector = np.asarray(gaps)
        return float(multipliers @ vector), vector

    # Two Jet directions give the complete mixed curvature of this actual
    # local chart, including multiplier-dependent beam-frame terms.
    a = Jet2.variable(0., 0, 2)
    b = Jet2.variable(0., 1, 2)
    high, low, frame = port.pose_pair(state)
    center = high + low
    frame_jets = so3_exp([a*value+b*other for value, other in
                          zip(u_beam[start+3:start+6], v_beam[start+3:start+6])])
    current_frame = [[sum(frame_jets[i][k]*frame[k, j] for k in range(3))
                      for j in range(3)] for i in range(3)]
    center_jets = [Jet2.constant(float(center[i]), 2) +
                   a*float(u_beam[start+i]) + b*float(v_beam[start+i])
                   for i in range(3)]
    work = Jet2.constant(0., 2)
    for (indices, weights), offset, row in zip(shape_rows, offsets, range(4)):
        position = weights @ (nodes[indices] + q.reshape(-1, 6)[indices, :3])
        pu = weights @ u_shell.reshape(-1, 6)[indices, :3]
        pv = weights @ v_shell.reshape(-1, 6)[indices, :3]
        distance = [Jet2.constant(float(position[i]), 2) + a*float(pu[i]) +
                    b*float(pv[i]) - center_jets[i] for i in range(3)]
        gaps = matvec(transpose(current_frame), distance)
        for axis in range(3):
            work += float(multipliers[3*row+axis]) * (gaps[axis]-float(offset[axis]))

    h = 2e-4
    f00, gap = independent(0., 0.)
    fp, _ = independent(h, 0.)
    fm, _ = independent(-h, 0.)
    gp, _ = independent(0., h)
    gm, _ = independent(0., -h)
    fpp, _ = independent(h, h)
    fpm, _ = independent(h, -h)
    fmp, _ = independent(-h, h)
    fmm, _ = independent(-h, -h)
    first_error = max(abs(work.gradient[0]-(fp-fm)/(2*h)),
                      abs(work.gradient[1]-(gp-gm)/(2*h)))
    mixed_fd = (fpp-fpm-fmp+fmm)/(4*h*h)
    mixed_error = abs(work.hessian[0, 1]-mixed_fd)
    physical_forces = (frame @ multipliers.reshape(4, 3).T).T
    shell_positions = np.array([
        weights @ (nodes[indices] + q.reshape(-1, 6)[indices, :3])
        for indices, weights in shape_rows])
    beam_force = -physical_forces.sum(axis=0)
    beam_moment = -np.cross(shell_positions-center, physical_forces).sum(axis=0)
    force_balance = float(np.linalg.norm(physical_forces.sum(axis=0)+beam_force))
    moment_balance = float(np.linalg.norm(
        np.cross(shell_positions, physical_forces).sum(axis=0) +
        np.cross(center, beam_force) + beam_moment))
    def physical_work(shell_direction, beam_direction):
        sampled = np.array([weights @ shell_direction.reshape(-1, 6)[indices, :3]
                            for indices, weights in shape_rows])
        return float(np.sum(physical_forces*sampled) +
                     beam_force@beam_direction[start:start+3] +
                     beam_moment@beam_direction[start+3:start+6])
    work_error = max(abs(work.gradient[0]-physical_work(u_shell, u_beam)),
                     abs(work.gradient[1]-physical_work(v_shell, v_beam)))
    from scipy.spatial.transform import Rotation
    motion = (Rotation.from_rotvec((.3, -.2, .4)).as_matrix(),
              np.array((2., -3., 1.)))
    moved, moved_gap = independent(0., 0., motion=motion)
    objective_error = max(abs(moved-f00), float(np.max(np.abs(moved_gap-gap))))
    if (abs(work.value-f00) > 1e-10 or first_error > 1e-6 or
            mixed_error > 1e-5 or objective_error > 1e-10 or
            abs(work.hessian[0, 1]) < 1e-3 or
            max(force_balance, moment_balance, work_error) > 1e-9):
        raise ArithmeticError("finite rigid footprint work/chart check failed")
    return {"potential_j": f00, "first_error_j": first_error,
            "mixed_hessian_j": float(work.hessian[0, 1]),
            "mixed_error_j": mixed_error,
            "objective_gap_work_error": objective_error,
            "force_balance_n": force_balance,
            "moment_balance_nm": moment_balance,
            "physical_work_error_j": work_error,
            "scope": "actual E4 position interpolation and native nodal Exp-chart, local nonzero-multiplier proof"}


def finite_distributing_work(shell: FlatE4TrialPanel, beam, port) -> dict:
    """Check the six-channel area-mean/polar patch through both finite charts."""
    panel = shell.panel
    shell_ref = pose_from_e4_total_strip(
        panel, .5, np.zeros(6*panel.node_count),
        half_length=.05, half_width=.05)
    base = beam.context.initial.mechanical
    beam_high, beam_low, beam_frame = port.pose_pair(base)
    joint = RigidPoseJoint(
        np.array((shell_ref[0], beam_high+beam_low)),
        np.array((shell_ref[1], beam_frame)))
    reference = joint.evaluate(
        np.array((shell_ref[0], beam_high)),
        np.array((shell_ref[1], beam_frame)), np.zeros(6),
        position_low=np.array((np.zeros(3), beam_low)))
    if np.max(np.abs(reference.constraints)) > 1e-12:
        raise ArithmeticError("distributing finite patch changed reference")
    rng = np.random.default_rng(615)
    shell_base = rng.normal(size=(panel.node_count, 6)) * .002
    first = rng.normal(size=shell_base.shape) * .025
    second = rng.normal(size=shell_base.shape) * .025
    beam_base = np.zeros(beam.context.count)
    beam_first = np.zeros_like(beam_base)
    beam_second = np.zeros_like(beam_base)
    start = 6*port.node_index
    beam_base[start:start+6] = (.0008, -.0005, .0004, .018, -.012, .009)
    beam_first[start:start+6] = (.003, -.002, .004, .07, -.05, .04)
    beam_second[start:start+6] = (-.002, .004, .001, -.06, .03, .05)
    state = beam.context.advance(base, beam_base)
    multipliers = np.array((18., -23., 31., 7., -100., 13.))
    analytic = directional_point_work(
        joint, panel, .5, shell_base.ravel(), first, second,
        port, state, beam_first, beam_second, multipliers,
        shell_port="physical_strip_005_005")

    def potential(a: float, b: float) -> float:
        shell_position, shell_frame = pose_from_e4_total_strip(
            panel, .5, (shell_base+a*first+b*second).ravel(),
            half_length=.05, half_width=.05)
        current = beam.context.advance(state, a*beam_first+b*beam_second)
        high, low, frame = port.pose_pair(current)
        return float(joint.evaluate(
            np.array((shell_position, high)),
            np.array((shell_frame, frame)), multipliers,
            position_low=np.array((np.zeros(3), low))).potential)

    h = 1e-5
    first_error = max(
        abs(analytic.first-(potential(h, 0.)-potential(-h, 0.))/(2*h)),
        abs(analytic.second-(potential(0., h)-potential(0., -h))/(2*h)))
    h = 2e-3
    mixed_fd = (potential(h, h)-potential(h, -h)-
                potential(-h, h)+potential(-h, -h))/(4*h*h)
    mixed_error = abs(analytic.mixed-mixed_fd)
    shell_position, shell_frame = pose_from_e4_total_strip(
        panel, .5, shell_base.ravel(), half_length=.05, half_width=.05)
    shell_d = directional_pose_from_e4_total_strip(
        panel, .5, shell_base.ravel(), first, second,
        half_length=.05, half_width=.05)
    high, low, beam_frame = port.pose_pair(state)
    beam_position = high + low
    evaluation = joint.evaluate(
        np.array((shell_position, high)),
        np.array((shell_frame, beam_frame)), multipliers,
        position_low=np.array((np.zeros(3), low)))
    shell_force = evaluation.residual[:3]
    shell_moment = evaluation.residual[3:6]
    beam_force = evaluation.residual[6:9]
    beam_moment = evaluation.residual[9:12]
    force_balance = float(np.linalg.norm(shell_force+beam_force))
    moment_balance = float(np.linalg.norm(
        np.cross(shell_position, shell_force)+shell_moment+
        np.cross(beam_position, beam_force)+beam_moment))

    def angular(frame_first: np.ndarray) -> np.ndarray:
        matrix = frame_first @ shell_frame.T
        return np.array((matrix[2, 1]-matrix[1, 2],
                         matrix[0, 2]-matrix[2, 0],
                         matrix[1, 0]-matrix[0, 1]))/2.

    beam_trace = port.tangent(state)
    first_trace = np.r_[shell_d.position_first,
                        angular(shell_d.frame_first),
                        beam_trace@beam_first]
    second_trace = np.r_[shell_d.position_second,
                         angular(shell_d.frame_second),
                         beam_trace@beam_second]
    physical_work_error = max(
        abs(analytic.first-evaluation.residual[:12]@first_trace),
        abs(analytic.second-evaluation.residual[:12]@second_trace))
    if (abs(analytic.potential-potential(0., 0.)) > 1e-10 or
            first_error > 1e-6 or mixed_error > 1e-5 or
            abs(analytic.mixed) < 1e-3 or
            max(force_balance, moment_balance, physical_work_error) > 1e-9):
        raise ArithmeticError("finite distributing patch pullback failed")
    return {"potential_j": analytic.potential,
            "first_error_j": first_error, "mixed_hessian_j": analytic.mixed,
            "mixed_error_j": mixed_error,
            "force_balance_n": force_balance,
            "moment_balance_nm": moment_balance,
            "physical_work_error_j": physical_work_error,
            "scope": "actual E4 area-mean/polar pose and native nodal Exp-chart, nonzero-multiplier local proof"}


def termination_local(shell: FlatE4TrialPanel, beam, *,
                      recover_interface: bool = False) -> dict:
    """Check a declared last-0.1 m bonded patch at the native beam tip."""
    panel = shell.panel
    shell_size = 6*panel.node_count
    port = NativeNodalSpatialPort(beam.context, beam.n_half_cells)
    state = beam.context.initial.mechanical
    high, low, frame = port.pose_pair(state)
    center = high+low
    station = .95
    points = tuple((x, y, 0.) for x, y in
                   ((.9, -.05), (1., -.05), (1., .05), (.9, .05)))
    if np.max(np.abs(center - (1., 0., .085))) > 1e-14:
        raise ArithmeticError("native tip footprint geometry changed")
    total = shell_size+beam.context.count
    beam_trace = port.tangent(state)
    rigid = np.zeros((12, total))
    offsets = []
    for i, point in enumerate(points):
        indices, shape = panel._shape(point[0], point[1])
        for node, weight in zip(indices, shape):
            rigid[3*i:3*i+3, 6*node:6*node+3] += weight*np.eye(3)
        rigid[3*i:3*i+3, shell_size:] = np.column_stack((
            -np.eye(3), skew(np.asarray(point)-center)))@beam_trace
        offsets.append(frame.T@(np.asarray(point)-center))
    shell_position, shell_frame = pose_from_e4_total_strip(
        panel, station, np.zeros(shell_size),
        half_length=.05, half_width=.05)
    joint = RigidPoseJoint(
        np.array((shell_position, center)),
        np.array((shell_frame, frame)))
    evaluation = joint.evaluate(
        np.array((shell_position, high)),
        np.array((shell_frame, frame)), np.zeros(6),
        position_low=np.array((np.zeros(3), low)))
    trace = reference_spatial_trace_strip(
        panel, station, half_length=.05, half_width=.05)
    distributing = np.column_stack((
        evaluation.constraint_jacobian[:, :6]@trace,
        evaluation.constraint_jacobian[:, 6:]@beam_trace))
    fixed_shell = {6*j*(panel.x_interval_count+1)+axis
                   for j in range(panel.n_y+1) for axis in range(6)}
    fixed_beam = {shell_size+index for index in beam.context.fixed}
    free = np.array([index for index in range(total)
                     if index not in fixed_shell and index not in fixed_beam])
    ranks = (int(np.linalg.matrix_rank(distributing[:, free])),
             int(np.linalg.matrix_rank(rigid[:, free])))
    # The patch lies entirely on the plate; its mean is not the native tip.
    # A physical shell My must still be recovered about one common origin.
    dmoment = np.zeros(6)
    dmoment[4] = -100.
    dx = np.array([point[0]-station for point in points])
    rmoment = np.zeros(12)
    rmoment[2::3] = -100.*dx/(dx@dx)
    reference = reference_nodes(panel)
    def shell_my(force):
        nodal = force[:shell_size].reshape(-1, 6)
        return float(np.sum(nodal[:, 4]-reference[:, 0]*nodal[:, 2]))
    moments = (shell_my(distributing.T@dmoment),
               shell_my(rigid.T@rmoment))
    rigid_reference_gap = max(float(np.linalg.norm(
        frame.T@(np.asarray(point)-center)-offset))
        for point, offset in zip(points, offsets))
    if (ranks != (6, 12) or
            max(abs(value-100.) for value in moments) > 1e-8 or
            max(rigid_reference_gap,
                float(np.max(np.abs(evaluation.constraints)))) > 1e-12):
        raise ArithmeticError("local termination footprint admission failed")

    # The small dense system is a bounded, independent G0 assembly oracle.
    # Both laws use identical E4/native-B3 tangents, supports and tip loads.
    shell_k = shell.evaluate(np.zeros(shell_size), station).tangent
    primal = linalg.block_diag(shell_k, beam.tangent.toarray())
    force_scale = 1e6
    displacement_scale = force_scale/(210e9*.07)
    nodal_scale = np.tile(np.full(6, displacement_scale),
                          panel.node_count+beam.context.nodal_count//6)
    resultants = np.r_[np.tile(SECTION_DIAGONAL[:3]*displacement_scale, 2),
                       np.tile(SECTION_DIAGONAL[3:]*displacement_scale, 4)]
    retained_scale = np.tile(np.r_[np.full(6, displacement_scale), resultants],
                              beam.n_half_cells//2)
    primal_scale = np.r_[nodal_scale, retained_scale]
    if len(primal_scale) != total:
        raise ArithmeticError("termination retained scale layout changed")

    def loaded(constraint: np.ndarray, axis: int) -> dict:
        m = len(constraint)
        n = len(free)
        rows = constraint[:, free]
        kkt = np.block([[primal[np.ix_(free, free)], rows.T],
                        [rows, np.zeros((m, m))]])
        scale = np.r_[primal_scale[free], np.full(m, force_scale)]
        energy = force_scale*displacement_scale
        scaled = (scale[:, None]*kkt)*scale[None, :]/energy
        load = np.zeros(total)
        load[shell_size+6*beam.n_half_cells+axis] = (
            100. if axis == 3 else 1000.)
        rhs = np.r_[load[free]*scale[:n]/energy, np.zeros(m)]
        answer = linalg.solve(scaled, rhs, assume_a="gen")
        full = np.zeros(total)
        full[free] = scale[:n]*answer[:n]
        multiplier = scale[n:]*answer[n:]
        residual = primal@full + constraint.T@multiplier-load
        free_error = float(np.linalg.norm(residual[free]) /
                           max(1., np.linalg.norm(load[free])))
        gap_error = float(np.linalg.norm(constraint@full))
        work_error = float(abs(full@(primal@full)-load@full))
        shell_reactions = residual[:shell_size].reshape(-1, 6)
        beam_reactions = residual[shell_size:shell_size+beam.context.nodal_count].reshape(-1, 6)
        shell_force = shell_reactions.sum(axis=0)[:3]
        beam_force = beam_reactions.sum(axis=0)[:3]
        support_force = shell_force+beam_force
        shell_nodes = reference_nodes(panel)
        beam_nodes = np.array([(i/beam.n_half_cells, 0., .085)
                               for i in range(beam.n_half_cells+1)])
        shell_moment = (np.cross(shell_nodes,shell_reactions[:,:3]).sum(axis=0)+
                        shell_reactions[:,3:].sum(axis=0))
        beam_moment = (np.cross(beam_nodes,beam_reactions[:,:3]).sum(axis=0)+
                       beam_reactions[:,3:].sum(axis=0))
        support_moment = shell_moment+beam_moment
        applied_force = load[shell_size+6*beam.n_half_cells:
                             shell_size+6*beam.n_half_cells+3]
        applied_moment = load[shell_size+6*beam.n_half_cells+3:
                              shell_size+6*beam.n_half_cells+6]
        balance_force = float(np.linalg.norm(support_force+applied_force))
        balance_moment = float(np.linalg.norm(support_moment+
            np.cross(center, applied_force)+applied_moment))
        if (max(free_error, gap_error, work_error) > 1e-7 or
                max(balance_force, balance_moment) > 1e-7 or
                not np.isfinite(full).all()):
            raise ArithmeticError("loaded local termination solution failed")
        result = {"tip_rx_rad": float(full[shell_size+6*beam.n_half_cells+3]),
                "tip_uz_m": float(full[shell_size+6*beam.n_half_cells+2]),
                "beam_section_rx_at_stations_rad":{
                    str(station):float(full[shell_size+6*(beam.n_half_cells*station//1000)+3])
                    for station in (0,250,500,750,1000)},
                "shell_support_wrench_N_Nm":[*shell_force.tolist(),
                                             *shell_moment.tolist()],
                "beam_root_wrench_N_Nm":[*beam_force.tolist(),
                                           *beam_moment.tolist()],
                "scaled_equilibrium_residual": free_error,
                "weak_gap_norm": gap_error,
                "work_error_j": work_error,
                "global_force_balance_n": balance_force,
                "global_moment_balance_nm": balance_moment}
        if recover_interface:
            # Kq + C.T lambda = applied. The load exerted by the other
            # domain on each component is therefore -C_component.T lambda.
            interface = -constraint.T@multiplier
            shell_interface=interface[:shell_size].reshape(-1,6)
            beam_interface=interface[shell_size:shell_size+
                                     beam.context.nodal_count].reshape(-1,6)
            def wrench(nodal, locations):
                return np.r_[nodal[:,:3].sum(axis=0),
                             (np.cross(locations,nodal[:,:3])+
                              nodal[:,3:]).sum(axis=0)]
            shell_wrench=wrench(shell_interface,shell_nodes)
            beam_wrench=wrench(beam_interface,beam_nodes)
            action_error=float(np.linalg.norm(shell_wrench+beam_wrench))
            virtual_work_error=float(abs(interface@full))
            if max(action_error,virtual_work_error)>1e-7:
                raise ArithmeticError("native interface wrench/work imbalance")
            active=np.flatnonzero(np.max(np.abs(shell_interface),axis=1)>1e-10)
            result["interface"]={
                "meaning":"forces/moments exerted by beam on shell and shell on beam; global-reference wrench; linearized accepted-reference KKT",
                "shell_wrench_N_Nm":shell_wrench.tolist(),
                "beam_wrench_N_Nm":beam_wrench.tolist(),
                "opposite_wrench_error":action_error,
                "zero_gap_virtual_work_error_j":virtual_work_error,
                "shell_work_j":float(interface[:shell_size]@full[:shell_size]),
                "beam_work_j":float(interface[shell_size:]@full[shell_size:]),
                "active_shell_nodes":[
                    {"node_row":int(i),"xyz_m":shell_nodes[i].tolist(),
                     "force_moment_N_Nm":shell_interface[i].tolist()}
                    for i in active],
                "beam_tip_force_moment_N_Nm":beam_interface[-1].tolist(),
                "retained_interface_norm":float(np.linalg.norm(
                    interface[shell_size+beam.context.nodal_count:])),
            }
        return result

    loaded_results = {
        family: {"torque_mx_100nm": loaded(constraint, 3),
                 "shear_fz_1000n": loaded(constraint, 2)}
        for family, constraint in (("distributing", distributing),
                                   ("rigid", rigid))}
    return {"patch_x_m": [.9, 1.], "beam_tip_x_m": 1.,
            "rows_and_supported_ranks": {"distributing": [6, ranks[0]],
                                         "rigid": [12, ranks[1]]},
            "reference_gap_max": max(
                rigid_reference_gap,
                float(np.max(np.abs(evaluation.constraints)))),
            "shell_my_nm": {"distributing": moments[0], "rigid": moments[1]},
            "loaded_linear_reference": loaded_results,
            "scope": "bounded linearized tip-patch dense oracle; no nonlinear termination qualification"}


def case(topology: str) -> dict:
    """Compare actual linearized local maps and their virtual work."""
    shell = FlatE4TrialPanel(make_shell(4, topology, n_y=4))
    beam = build(4, z_offset=.085,
                 torsion_constant=square_torsion_constant())
    port = NativeNodalSpatialPort(beam.context, 2)
    shell_size = 6 * shell.panel.node_count
    count = shell_size + beam.context.count
    points = tuple((.5+dx, dy, 0.) for dx, dy in
                   ((-.05, -.05), (.05, -.05), (.05, .05), (-.05, .05)))
    beam_high, beam_low, beam_frame = port.pose_pair(
        beam.context.initial.mechanical)
    beam_center = beam_high + beam_low
    if not np.allclose(beam_center, (.5, 0., .085), atol=1e-14):
        raise ArithmeticError("native patch center changed")
    beam_trace = port.tangent(beam.context.initial.mechanical)
    rigid = np.zeros((12, count))
    for i, point in enumerate(points):
        indices, shape = shell.panel._shape(point[0], point[1])
        for node, weight in zip(indices, shape):
            rigid[3*i:3*i+3, 6*node:6*node+3] += weight * np.eye(3)
        rigid[3*i:3*i+3, shell_size:] = np.column_stack((
            -np.eye(3), skew(np.asarray(point)-beam_center))) @ beam_trace

    shell_position, shell_frame = pose_from_e4_total_strip(
        shell.panel, .5, np.zeros(shell_size),
        half_length=.05, half_width=.05)
    joint = RigidPoseJoint(
        np.array((shell_position, beam_center)),
        np.array((shell_frame, beam_frame)))
    evaluation = joint.evaluate(
        np.array((shell_position, beam_high)),
        np.array((shell_frame, beam_frame)), np.zeros(6),
        position_low=np.array((np.zeros(3), beam_low)))
    if np.linalg.norm(evaluation.constraints) > 1e-12:
        raise ArithmeticError("distributing patch is not stress free")
    trace = reference_spatial_trace_strip(
        shell.panel, .5, half_length=.05, half_width=.05)
    distributing = np.column_stack((
        evaluation.constraint_jacobian[:, :6] @ trace,
        evaluation.constraint_jacobian[:, 6:] @ beam_trace))

    nx = shell.panel.x_interval_count
    fixed_shell = {6*j*(nx+1)+axis
                   for j in range(shell.panel.n_y+1)
                   for axis in range(6)}
    fixed_beam = {shell_size+index for index in beam.context.fixed}
    free = np.array([index for index in range(count)
                     if index not in fixed_shell and index not in fixed_beam])
    ranks = (int(np.linalg.matrix_rank(distributing[:, free])),
             int(np.linalg.matrix_rank(rigid[:, free])))
    if ranks != (6, 12):
        raise ArithmeticError(f"local footprint rows lost rank: {ranks}")

    reference = reference_nodes(shell.panel)
    translation = np.array((.003, -.002, .004))
    rotation = np.array((.008, -.006, .01))
    common = np.zeros(count)
    shell_nodal = common[:shell_size].reshape(-1, 6)
    shell_nodal[:, :3] = translation + np.cross(rotation, reference)
    shell_nodal[:, 3:] = rotation
    for node in range(beam.n_half_cells+1):
        origin = np.array((node/beam.n_half_cells, 0., .085))
        start = shell_size + 6*node
        common[start:start+3] = translation + np.cross(rotation, origin)
        common[start+3:start+6] = rotation
    rigid_gap = float(np.max(np.abs(rigid @ common)))
    distributing_gap = float(np.max(np.abs(distributing @ common)))
    if max(rigid_gap, distributing_gap) > 2e-12:
        raise ArithmeticError("local footprint lost rigid-motion invariance")

    # This field vanishes at the fixed left edge and has zero strip mean.
    # Its geometric z slope is nonzero while independent shell rotations stay zero.
    mean, _ = strip_weights(shell.panel, .5,
                            half_length=.05, half_width=.05)
    x = reference[:, 0]
    a = (x-.5)*(x+.5)
    b = x+.5
    shear = np.zeros(count)
    shear[:shell_size].reshape(-1, 6)[:, 2] = .01 * (
        a - (mean@a)/(mean@b)*b)
    energy = float(shear[:shell_size] @ (
        physical_shell_stiffness(shell.panel) @ shear[:shell_size]))
    distribution_shear_gap = float(np.max(np.abs(distributing @ shear)))
    rigid_shear_gap = float(np.linalg.norm(rigid @ shear))
    if (energy <= 0. or distribution_shear_gap > 2e-12 or
            rigid_shear_gap <= 1e-5):
        raise ArithmeticError("physical transverse-shear distinction failed")

    # Paired +My multipliers act on the same shell patch and beam station.
    distributing_moment = np.zeros(6)
    # c_R = Log(R_shell^T R_beam), so a negative multiplier gives +My
    # on the shell. The corner-force construction below gives that same
    # physical shell moment through its eccentric Fz resultants.
    distributing_moment[4] = -100.
    dx = np.array([point[0]-.5 for point in points])
    rigid_moment = np.zeros(12)
    rigid_moment[2::3] = -100.*dx/(dx@dx)
    distributed_force = distributing.T @ distributing_moment
    rigid_force = rigid.T @ rigid_moment
    def shell_my(force: np.ndarray) -> float:
        nodal = force[:shell_size].reshape(-1, 6)
        return float(np.sum(nodal[:, 4] - reference[:, 0]*nodal[:, 2]))
    shell_moments = (shell_my(distributed_force), shell_my(rigid_force))
    if max(abs(value-100.) for value in shell_moments) > 1e-8:
        raise ArithmeticError(f"paired shell moments differ: {shell_moments}")
    distributed_shear_work = float(distributed_force @ shear)
    rigid_shear_work = float(rigid_force @ shear)
    if (abs(distributed_shear_work) > 1e-9 or
            abs(rigid_shear_work) < 1e-3 or
            max(abs(distributed_force @ common),
                abs(rigid_force @ common)) > 1e-9):
        raise ArithmeticError("local moment work distinction failed")
    return {
        "topology": topology, "patch_half_widths_m": [.05, .05],
        "beam_axis_z_m": .085, "native_half_cells": 4,
        "rows_and_supported_ranks": {"distributing": [6, ranks[0]],
                                     "rigid": [12, ranks[1]]},
        "common_rigid_gap_max_m_rad": {
            "distributing": distributing_gap, "rigid": rigid_gap},
        "supported_shell_transverse_shear_mode": {
            "physical_energy_twice_j": energy,
            "distributing_gap_max_m_rad": distribution_shear_gap,
            "rigid_gap_l2_m": rigid_shear_gap,
            "distributing_my_work_j": distributed_shear_work,
            "rigid_my_work_j": rigid_shear_work},
        "shell_drilling_force_max": {
            "distributing": float(np.max(np.abs(
                distributed_force[:shell_size][5::6]))),
            "rigid": float(np.max(np.abs(
                rigid_force[:shell_size][5::6])))},
        "paired_shell_my_nm": {"distributing": shell_moments[0],
                               "rigid": shell_moments[1]},
        "finite_rigid_chart": finite_rigid_work(
            shell, beam, port, points, rigid_moment),
        "finite_distributing_chart": finite_distributing_work(
            shell, beam, port),
        "tip_termination_local": termination_local(shell, beam),
        "scope": "local patch work and bounded linearized tip KKT; rigid patch intentionally restrains local shear; not a true-line or production result",
    }


def run(output_dir: Path) -> None:
    source = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                     text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"],
                               text=True).strip():
        raise RuntimeError("run requires clean source")
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "EXPLORATORY_NOT_G0_ACCEPTANCE", "source_sha": source,
        "question": "Do distinct finite-width laws remain consistent and give distinguishable, balanced linearized responses for the same last-0.1 m beam-tip patch?",
        "decision": "Failure of reference, rank, equilibrium, work or global balance requires map/assembly revision. Passing bounded loaded responses keeps both declared idealizations for physical-reference and nonlinear checks; it does not select a true-line multiplier space.",
        "python": platform.python_version(), "numpy": np.__version__,
        "threads": 1, "models": ["Q4", "S3-V2D"],
        "geometry": "plate [-.5,1.5]x[-.5,.5] z=0 thickness .07 m; center patch x=.5±.05 and terminal patch x=[.9,1.0], y±.05; B3 axis z=.085 m, x=[0,1]",
        "section": "square-bar Saint-Venant J; other native generalized resultants unchanged",
        "support": "full shell left edge and native B3 root fixed",
        "loads": "separate +100 Nm native tip Mx and +1000 N native tip Fz for the bounded terminal KKT",
        "scope": "local work and dense linearized terminal KKT; no nonlinear, convergence, physical-accuracy, production-footprint or true-line qualification",
    }
    with threadpool_limits(limits=1):
        manifest["effective_threadpools"] = threadpool_info()
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n")
        with (output_dir / "results.jsonl").open("w") as stream:
            for topology in ("Q4", "S3-V2D"):
                result = case(topology)
                stream.write(json.dumps(result, allow_nan=False) + "\n")
                stream.flush()
                print(topology, result["supported_shell_transverse_shear_mode"],
                      flush=True)


def run_terminal_beam_resolution(output_dir: Path) -> None:
    """One-axis native beam check with the E4 panel and footprint fixed."""
    source = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                     text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"],
                               text=True).strip():
        raise RuntimeError("run requires clean source")
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "EXPLORATORY_NOT_G0_ACCEPTANCE",
        "source_sha": source,
        "question": "Does native B3 half-cell resolution cause the rigid/distributing terminal response difference on one fixed E4 Q4 panel?",
        "decision": "Material drift with beam 4->8->16 requires beam/port convergence work before comparing footprint laws; roundoff agreement shifts the remaining discrepancy to shell or connection idealization. Neither selects a production law.",
        "panel": "Q4 x4/y4, t=.07 m; full left-edge clamp",
        "beam": "native B3 half-cells 4/8/16, centroid z=.085 m, square Saint-Venant J; root fixed",
        "footprint": "last 0.1 m, y +/-0.05 m; same rigid and distributing laws",
        "loads": "separate +100 Nm tip Mx and +1000 N tip Fz",
        "python": platform.python_version(), "numpy": np.__version__,
        "scope": "bounded linearized fixed-panel check; no external accuracy, mesh convergence or nonlinear qualification",
    }
    with threadpool_limits(limits=1):
        manifest["effective_threadpools"] = threadpool_info()
        (output_dir/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        shell = FlatE4TrialPanel(make_shell(4, "Q4", n_y=4))
        records=[]
        for half in (4, 8, 16):
            beam = build(half, z_offset=.085,
                         torsion_constant=square_torsion_constant())
            result=termination_local(shell, beam)
            records.append({"beam_half_cells":half, "response":result})
            print(half, json.dumps(result["loaded_linear_reference"]), flush=True)
        (output_dir/"results.json").write_text(json.dumps(records, indent=2,
                                             allow_nan=False)+"\n")


def run_terminal_shell_resolution(output_dir: Path) -> None:
    """Bounded native terminal-law response under simultaneous shell refinement."""
    source = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                     text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"],
                               text=True).strip():
        raise RuntimeError("run requires clean source")
    output_dir.mkdir(parents=True, exist_ok=False)
    levels=(2,4,8)
    manifest={
        "status":"EXPLORATORY_NOT_G0_ACCEPTANCE",
        "source_sha":source,
        "question":"Do native six-channel terminal responses, support-wrench partitions and beam-section rotation profiles approach the section-resolved external terminal reference as E4 refines?",
        "decision":"A strong native mesh trend bars a law ranking. Similar support partition with a near-patch rotation difference directs investigation to local connection/warping; a span-wide difference directs it to section or shell stiffness. Neither outcome selects a footprint or true-line multiplier space.",
        "shell_formulations":["Q4","S3-V2D"],
        "shell_levels_x_half_y":list(levels),
        "beam_half_cells":8,
        "geometry":"plate [-.5,1.5]x[-.5,.5] m, thickness .07 m; B3 straight x=[0,1] m, centroid z=.085 m; patch x=[.9,1], y=+/- .05 m",
        "supports":"full E4 left edge and B3 root fixed",
        "loads":"separate +100 N m tip Mx and +1000 N tip Fz",
        "external_context":"C3D20R solid terminal patch has a declared 2 micrometre clearance outside the patch; native straight beam does not",
        "limits":{"max_dense_kkt_rows":2300,"blas_threads":1},
        "scope":"linearized dense G0 native response screen; no production assembly, nonlinear qualification, exact solid geometry parity or true-line selection",
        "python":platform.python_version(),"numpy":np.__version__,
    }
    with threadpool_limits(limits=1):
        manifest["effective_threadpools"]=threadpool_info()
        (output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        beam=build(8,z_offset=.085,torsion_constant=square_torsion_constant())
        records=[]
        for topology in manifest["shell_formulations"]:
            for level in levels:
                started=time.perf_counter()
                shell=FlatE4TrialPanel(make_shell(level,topology,n_y=level))
                rows=6*shell.panel.node_count+beam.context.count+12
                if rows>manifest["limits"]["max_dense_kkt_rows"]:
                    raise ValueError("terminal shell-resolution dense row cap exceeded")
                result=termination_local(shell,beam)
                record={"topology":topology,"shell_level":level,
                        "shell_nodes":shell.panel.node_count,
                        "dense_kkt_rows_bound":rows,
                        "seconds":time.perf_counter()-started,
                        "response":result}
                records.append(record)
                (output_dir/"results.json").write_text(json.dumps(
                    records,indent=2,allow_nan=False)+"\n")
                print(topology,level,json.dumps(result["loaded_linear_reference"]),
                      flush=True)


def run_terminal_interface(output_dir: Path) -> None:
    """Recover actual native KKT interface actions for reference design."""
    source=subprocess.check_output(["git","rev-parse","HEAD"],
                                   text=True).strip()
    if subprocess.check_output(["git","status","--porcelain"],
                               text=True).strip():
        raise RuntimeError("run requires clean source")
    output_dir.mkdir(parents=True,exist_ok=False)
    manifest={
        "status":"EXPLORATORY_NOT_G0_ACCEPTANCE",
        "source_sha":source,
        "question":"What shell force/moment fields do the actual native terminal KKT multipliers transmit under separate torque and transverse-force loading?",
        "decision":"Use the recovered shape, work and wrench to design a physically matched external bond comparison. If the fields differ materially between laws, smooth plate-only traction cannot rank them. This does not select a connection.",
        "geometry":"Q4 plate [-.5,1.5]x[-.5,.5] m, t=.07 m; last-0.1 m square footprint; B3 centroid z=.085 m",
        "model":"E4 Q4 x4/y4, native generalized B3 eight half-cells, retained coordinates, distributing six rows or rigid twelve rows",
        "support":"full shell left-edge and B3 root fixed",
        "loads":"separate +100 Nm tip Mx and +1000 N tip Fz",
        "limits":{"max_dense_kkt_rows":1000,"blas_threads":1},
        "scope":"linearized native coupled interface recovery; no continuum contact, external accuracy, footprint or line qualification",
        "python":platform.python_version(),"numpy":np.__version__,
    }
    with threadpool_limits(limits=1):
        manifest["effective_threadpools"]=threadpool_info()
        (output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        shell=FlatE4TrialPanel(make_shell(4,"Q4",n_y=4))
        beam=build(8,z_offset=.085,
                   torsion_constant=square_torsion_constant())
        rows=6*shell.panel.node_count+beam.context.count+12
        if rows>manifest["limits"]["max_dense_kkt_rows"]:
            raise ValueError("terminal interface dense row cap exceeded")
        result=termination_local(shell,beam,recover_interface=True)
        (output_dir/"results.json").write_text(json.dumps(
            result,indent=2,allow_nan=False)+"\n")
        for law,cases in result["loaded_linear_reference"].items():
            for loading,case in cases.items():
                field=case["interface"]
                print(json.dumps({"law":law,"loading":loading,
                                  "shell_wrench_N_Nm":field["shell_wrench_N_Nm"],
                                  "active_shell_nodes":len(field["active_shell_nodes"]),
                                  "shell_work_j":field["shell_work_j"]}),
                      flush=True)


def graded_terminal_coupling(patch_cells: int, beam, *,
                             plate_thickness: float = .07,
                             transverse_cells: int | None = None,
                             include_surface_moments: bool = False,
                             include_top_face_two_intervals: bool = False,
                             top_face_intervals: int | tuple[int, ...] | None = None,
                             top_face_only: bool = False,
                             recover_top_face_local: bool = False,
                             include_line_hybrid: bool = False,
                             include_line_hybrid_segmented: bool = False,
                             include_line_hybrid_spaces: bool = False,
                             include_line_hybrid_ablation: bool = False,
                             sparse_line_only: bool = False,
                             line_quadrature_order: int = 3,
                             line_multiplier_intervals: int = 2,
                             mixed_s3: bool = False,
                             mixed_s3_diagonal: str = "bl_tr",
                             force_sign: float = 1.,
                             include_tip_my: bool = False,
                             include_pitch_ablation: bool = False) -> dict:
    """Linearized native B3/E4 terminal KKT on a resolved local grid."""
    if plate_thickness not in (.07,.035):
        raise ValueError("bounded native plate thickness required")
    if force_sign not in (-1., 1.):
        raise ValueError("declared unit-sign terminal force required")
    if include_tip_my and not (
            sparse_line_only and include_line_hybrid_spaces and
            patch_cells == 4 and transverse_cells == 4 and
            plate_thickness == .035 and beam.n_half_cells == 12 and
            not mixed_s3):
        raise ValueError("loaded pitch screen requires bounded sparse C1 line")
    if include_pitch_ablation and not include_tip_my:
        raise ValueError("pitch ablation requires the bounded loaded My screen")
    if include_top_face_two_intervals and top_face_intervals is not None:
        raise ValueError("choose one top-face interval selector")
    top_face_counts = ((2,) if include_top_face_two_intervals else
                       (top_face_intervals,) if type(top_face_intervals) is int else
                       top_face_intervals if type(top_face_intervals) is tuple else
                       ())
    if top_face_counts and not (
            top_face_counts in ((2,), (4,), (8,), (2, 4), (4, 8)) and
            patch_cells in (4, 8, 16) and
            all(patch_cells >= 2*count and patch_cells % count == 0
                for count in top_face_counts) and transverse_cells == 4 and
            plate_thickness == .035 and beam.n_half_cells in (12, 24) and
            beam.model.name.endswith("-c1_tangent_2um")):
        raise ValueError("top-face screen requires two shell hosts per interval")
    if top_face_intervals is not None and not top_face_counts:
        raise ValueError("top-face interval count must be 2, 4, 8, (2, 4) or (4, 8)")
    if top_face_only and not top_face_counts:
        raise ValueError("top-face-only screen requires its declared candidate")
    if top_face_only and sparse_line_only:
        raise ValueError("choose one bounded sparse candidate route")
    if recover_top_face_local and not (
            top_face_only and top_face_counts == (4, 8) and
            patch_cells == 16 and not mixed_s3):
        raise ValueError("selected local recovery requires x16 top-face four/eight Q4 pair")
    sparse_selected_only = sparse_line_only or top_face_only
    if include_line_hybrid_spaces and plate_thickness != .035:
        raise ValueError("weighted C1 line spaces require the 35 mm plate")
    if (type(line_multiplier_intervals) is not int or
            line_multiplier_intervals not in (2, 4, 8, 16) or
            (line_multiplier_intervals != 2 and
             (not include_line_hybrid_spaces or
              include_line_hybrid_segmented or
              patch_cells % line_multiplier_intervals != 0))):
        raise ValueError("multiplier refinement requires aligned bounded weighted line spaces")
    if sparse_line_only and (not include_line_hybrid_spaces or
                             patch_cells not in (2, 4, 8, 16) or
                             (patch_cells == 2 and transverse_cells != 4)):
        raise ValueError("sparse line comparison requires bounded weighted C1 case")
    if line_quadrature_order not in (3, 5) or (
            line_quadrature_order != 3 and not sparse_line_only):
        raise ValueError("fifth-order quadrature requires bounded sparse line comparison")
    if mixed_s3 and not (sparse_line_only and patch_cells == 2 and
                         transverse_cells == 4 and
                         plate_thickness == .035 and
                         beam.n_half_cells == 12 and
                         not (include_line_hybrid or
                              include_line_hybrid_segmented or
                              include_line_hybrid_ablation or
                              include_surface_moments)):
        raise ValueError("mixed C1 Q4/S3 requires bounded sparse weighted line case")
    if (mixed_s3_diagonal not in ("bl_tr", "br_tl") or
            not mixed_s3 and mixed_s3_diagonal != "bl_tr"):
        raise ValueError("mixed C1 S3 diagonal requires bounded mixed case")
    if include_line_hybrid_ablation and patch_cells != 2:
        raise ValueError("bounded hybrid ablation requires the fixed C1 two-cell grid")
    beam_axis_z=plate_thickness/2+.05
    xs,ys=local_grid(patch_cells,transverse_cells=transverse_cells)
    if recover_top_face_local:
        shell_k,physical,shell_elements=raw_e4_tangent(
            xs,ys,thickness=plate_thickness,
            return_element_operators=True)
    else:
        shell_k,physical=(raw_e4_mixed_c1_tangent(xs,ys,
                            thickness=plate_thickness,
                            diagonal=mixed_s3_diagonal) if mixed_s3 else
                          raw_e4_tangent(xs,ys,thickness=plate_thickness))
        shell_elements=None
    s3_cells=c1_mixed_s3_cells(xs,ys) if mixed_s3 else frozenset()
    shell_nodes=np.array([(x,y,0.) for y in ys for x in xs])
    shell_size=6*len(shell_nodes)
    total=shell_size+beam.context.count
    trace=structured_q4_patch_trace(xs,ys)
    port=NativeNodalSpatialPort(beam.context,beam.n_half_cells)
    base=beam.context.initial.mechanical
    high,low,frame=port.pose_pair(base)
    center=high+low
    shell_center=trace[0,0::6]@shell_nodes
    if (np.linalg.norm(shell_center-(.95,0.,0.))>1e-12 or
            np.linalg.norm(center-(1.,0.,beam_axis_z))>1e-12):
        raise ArithmeticError("graded terminal reference geometry changed")
    joint=RigidPoseJoint(np.array((shell_center,center)),
                         np.array((np.eye(3),frame)))
    evaluation=joint.evaluate(
        np.array((shell_center,high)),np.array((np.eye(3),frame)),
        np.zeros(6),position_low=np.array((np.zeros(3),low)))
    beam_trace=port.tangent(base)
    distributing=np.column_stack((
        evaluation.constraint_jacobian[:,:6]@trace,
        evaluation.constraint_jacobian[:,6:]@beam_trace))
    points=((.9,-.05), (1.,-.05), (1.,.05), (.9,.05))
    rigid=np.zeros((12,total))
    for point_index,(x,y) in enumerate(points):
        nodes,values=structured_q4_point_shape(xs,ys,x,y)
        for node,value in zip(nodes,values):
            rigid[3*point_index:3*point_index+3,
                  6*node:6*node+3]+=value*np.eye(3)
        point=np.array((x,y,0.))
        rigid[3*point_index:3*point_index+3,shell_size:]=(
            np.column_stack((-np.eye(3),skew(point-center)))@beam_trace)
    fixed_shell={6*j*len(xs)+axis for j in range(len(ys))
                 for axis in range(6)}
    fixed_beam={shell_size+index for index in beam.context.fixed}
    free=np.array([i for i in range(total)
                   if i not in fixed_shell and i not in fixed_beam])
    primal=(sparse.block_diag((sparse.csr_matrix(shell_k),beam.tangent),
                              format="csr") if sparse_selected_only else
            linalg.block_diag(shell_k,beam.tangent.toarray()))
    force_scale=1e6
    displacement_scale=force_scale/(210e9*plate_thickness)
    nodal_scale=np.tile(np.full(6,displacement_scale),
                        len(shell_nodes)+beam.context.nodal_count//6)
    resultants=np.r_[np.tile(SECTION_DIAGONAL[:3]*displacement_scale,2),
                     np.tile(SECTION_DIAGONAL[3:]*displacement_scale,4)]
    retained_scale=np.tile(np.r_[np.full(6,displacement_scale),resultants],
                            beam.n_half_cells//2)
    primal_scale=np.r_[nodal_scale,retained_scale]
    if len(primal_scale)!=total:
        raise ArithmeticError("graded retained scale layout changed")
    beam_nodes=(beam.context.initial.mechanical.positions+
                beam.context.initial.mechanical.position_low)
    stations=(0,250,500,750,900,950,1000)
    station_nodes={station:int(np.argmin(np.abs(
        beam_nodes[:,0]-station/1000))) for station in stations}
    required=(0,250,500,750,1000)
    if any(abs(beam_nodes[station_nodes[station],0]-station/1000)>1e-12
           for station in required):
        raise ValueError("terminal reference lacks required beam stations")
    has_local_stations=all(
        abs(beam_nodes[station_nodes[station],0]-station/1000)<1e-12
        for station in (900,950))
    if not has_local_stations:
        station_nodes={station:station_nodes[station] for station in required}
    def wrench(nodal,locations):
        return np.r_[nodal[:,:3].sum(axis=0),
                     (np.cross(locations,nodal[:,:3])+nodal[:,3:]).sum(axis=0)]
    def retained_beam_wrench(generalized):
        """Recover six physical resultants through common rigid variations."""
        basis=np.eye(3)
        result=np.zeros(6)
        for axis,direction in enumerate(basis):
            translation=np.zeros(beam.context.count)
            rotation=np.zeros(beam.context.count)
            for node,location in enumerate(beam_nodes):
                start=6*node
                translation[start:start+3]=direction
                rotation[start:start+3]=np.cross(direction,location)
                rotation[start+3:start+6]=direction
            for element in range(beam.n_half_cells//2):
                for cell in (0,1):
                    start=beam.context.nodal_count+24*element+3*cell
                    rotation[start:start+3]=direction
            result[axis]=generalized@translation
            result[3+axis]=generalized@rotation
        return result
    families=[("distributing",distributing),("rigid",rigid)]
    surface_diagnostics=None
    surface_patch=None
    line_gap_patch=None
    top_face_reference={}
    top_face_patches={}
    if include_surface_moments:
        from scripts.b3_e4_f03_distributed_surface_finite import (
            FiniteSurfaceMomentPatch,
        )
        surface,surface_diagnostics=distributed_surface_patch_trace(
            xs,ys,beam,plate_thickness=plate_thickness)
        surface_patch=FiniteSurfaceMomentPatch(
            xs,ys,beam,thickness=plate_thickness)
        families.append(("surface_moments",surface))
    if top_face_counts:
        from scripts.b3_e4_f03_distributed_surface_finite import (
            FiniteSurfaceTopFaceLine,
        )
        for top_face_count in top_face_counts:
            top_face_knots = np.linspace(.9, 1., top_face_count+1)
            top_face_pieces = [FiniteSurfaceTopFaceLine(
                xs, ys, beam, thickness=plate_thickness,
                x_interval=(float(low), float(high)))
                for low, high in zip(top_face_knots[:-1],
                                     top_face_knots[1:])]
            top_face_name = {2: "top_face_two_intervals",
                             4: "top_face_four_intervals",
                             8: "top_face_eight_intervals"}[top_face_count]
            reference_gaps = [piece.evaluate_e4_total(
                np.zeros(piece.shell_size), base)[:, 0]
                for piece in top_face_pieces]
            max_gap = float(max(np.max(np.abs(gap))
                                for gap in reference_gaps))
            if max_gap > 1e-12:
                raise ArithmeticError("top-face line opens at its reference")
            top_face_reference[top_face_name] = {
                "max_reference_gap_m_rad": max_gap,
                "basis_integrals_m": [piece.basis_integral_m
                                      for piece in top_face_pieces],
                "quadrature_points": [len(piece.points)
                                      for piece in top_face_pieces],
            }
            top_face_patches[top_face_name]=top_face_pieces
            families.append((top_face_name, np.vstack(
                [piece.reference_trace() for piece in top_face_pieces])))
    if (include_line_hybrid or include_line_hybrid_segmented or
            include_line_hybrid_spaces or include_line_hybrid_ablation):
        from scripts.b3_e4_f03_distributed_surface_finite import (
            FiniteSurfaceLineHybrid,
        )
        if include_line_hybrid_spaces:
            line_gap_patch=FiniteSurfaceLineHybrid(
                xs,ys,beam,thickness=plate_thickness,mixed_s3=mixed_s3,
                mixed_s3_diagonal=mixed_s3_diagonal)
        if include_line_hybrid or include_line_hybrid_ablation:
            hybrid = FiniteSurfaceLineHybrid(
                xs, ys, beam, thickness=plate_thickness)
            single_rows = hybrid.reference_trace()
            if include_line_hybrid:
                families.append(("line_hybrid_single_interval", single_rows))
        if include_line_hybrid_segmented or include_line_hybrid_ablation:
            segments = [FiniteSurfaceLineHybrid(
                xs, ys, beam, thickness=plate_thickness,
                x_interval=interval)
                for interval in ((.9, .95), (.95, 1.))]
            pieces = [segment.reference_trace() for segment in segments]
            if include_line_hybrid_segmented:
                families.append(("line_hybrid_two_intervals",
                                 np.vstack(pieces)))
            if include_line_hybrid_ablation:
                common, alternating = ((pieces[0]+pieces[1])/2,
                                       (pieces[0]-pieces[1])/2)
                if np.max(np.abs(common-single_rows)) > 1e-12:
                    raise ArithmeticError("C1 common interval changed")
                families.extend((
                    ("line_hybrid_common_alt_fz", np.vstack((
                        common, alternating[[2]]))),
                    ("line_hybrid_common_alt_fx", np.vstack((
                        common, alternating[[0]]))),
                    ("line_hybrid_common_alt_my", np.vstack((
                        common, alternating[[4]]))),
                    ("line_hybrid_common_alt_fx_my", np.vstack((
                        common, alternating[[0, 4]]))),
                    ("line_hybrid_common_alt_except_fz", np.vstack((
                        common, alternating[[0, 1, 3, 4, 5]]))),
                    ("line_hybrid_common_alt_fy_mx_mz", np.vstack((
                        common, alternating[[1, 3, 5]]))),
                ))
        space_masses={}
        if include_line_hybrid_spaces:
            from scripts.b3_e4_f03_distributed_surface_finite import (
                c1_hybrid_line_space,
            )
            for kind,label in (("piecewise_constant", "p0"),
                               ("continuous_linear", "p1"),
                               ("dual_linear", "dual_p1")):
                rows,masses,_=c1_hybrid_line_space(
                    xs,ys,beam,kind=kind,
                    intervals=line_multiplier_intervals,
                    thickness=plate_thickness,
                    quadrature_order=line_quadrature_order,
                    mixed_s3=mixed_s3,
                    mixed_s3_diagonal=mixed_s3_diagonal)
                name=f"line_hybrid_{label}_weighted"
                families.append((name,rows))
                space_masses[name]=masses.tolist()
            if include_line_hybrid_segmented and np.max(np.abs(
                    families[-3][1]-np.vstack(pieces))) > 1e-12:
                raise ArithmeticError("C1 weighted P0 changed existing rows")
    else:
        space_masses={}
    if top_face_only:
        families=[item for item in families
                  if item[0].startswith("top_face_")]
    if sparse_line_only:
        families=[item for item in families if item[0] in space_masses]
    if include_pitch_ablation:
        families += [(name+"_without_pitch",
                      np.delete(rows,np.arange(4,len(rows),6),axis=0))
                     for name,rows in families]
    records={}
    for family,constraint in families:
        rank=int(np.linalg.matrix_rank(constraint[:,free]))
        if rank!=len(constraint):
            raise ArithmeticError("graded terminal rows lost rank")
        m=len(constraint)
        n=len(free)
        rows=constraint[:,free]
        row_inventory = None
        if sparse_selected_only:
            # This is an algebraic representability check, not an inf-sup
            # or physical stability measure. Use the existing coordinate
            # scales and normalize each mixed-unit row before the split.
            scaled_rows = rows*primal_scale[free][None, :]
            row_norm = np.linalg.norm(scaled_rows, axis=1)
            if not np.isfinite(row_norm).all() or np.any(row_norm <= 0.):
                raise ArithmeticError("bounded line row scale is invalid")
            scaled_rows /= row_norm[:, None]
            shell_free = int(np.count_nonzero(free < shell_size))

            def inventory(part):
                singular = linalg.svdvals(part, check_finite=True)
                largest = float(singular[0])
                return {
                    "rank_at_relative_1e_minus_10": int(np.count_nonzero(
                        singular > 1e-10*largest)),
                    "rows": m,
                    "smallest_over_largest": (
                        float(singular[-1]/largest) if largest else 0.),
                }

            row_inventory = {
                "whole": inventory(scaled_rows),
                "shell_only": inventory(scaled_rows[:, :shell_free]),
                "beam_only": inventory(scaled_rows[:, shell_free:]),
            }
        if (n+m>1900 and not sparse_selected_only or
                sparse_selected_only and max(shell_size,beam.context.count)>1900):
            raise ValueError("graded terminal KKT exceeds bounded row cap")
        scale=np.r_[primal_scale[free],np.full(m,force_scale)]
        energy=force_scale*displacement_scale
        if sparse_selected_only:
            kkt=sparse.bmat(((primal[free][:,free],sparse.csr_matrix(rows.T)),
                             (sparse.csr_matrix(rows),None)),format="csc")
            diagonal=sparse.diags(scale,format="csc")
            solve=sparse.linalg.splu(diagonal@kkt@diagonal/energy).solve
        else:
            kkt=np.block([[primal[np.ix_(free,free)],rows.T],
                          [rows,np.zeros((m,m))]])
            scaled=(scale[:,None]*kkt)*scale[None,:]/energy
            lu,piv=linalg.lu_factor(scaled)
            solve=lambda rhs: linalg.lu_solve((lu,piv),rhs)
        cases={}
        load_cases = (("tip_mx",3,100.),
                      ("tip_fz",2,1000.*force_sign))
        if include_tip_my:
            load_cases += (("tip_my",4,100.),)
        for loading,axis,magnitude in load_cases:
            load=np.zeros(total)
            load[shell_size+6*beam.n_half_cells+axis]=magnitude
            rhs=np.r_[load[free]*scale[:n]/energy,np.zeros(m)]
            answer=solve(rhs)
            full=np.zeros(total)
            full[free]=scale[:n]*answer[:n]
            multiplier=scale[n:]*answer[n:]
            residual=primal@full+constraint.T@multiplier-load
            interface=-constraint.T@multiplier
            shell_interface=interface[:shell_size].reshape(-1,6)
            beam_interface=interface[shell_size:shell_size+
                                     beam.context.nodal_count].reshape(-1,6)
            shell_wrench=wrench(shell_interface,shell_nodes)
            beam_wrench=(retained_beam_wrench(interface[shell_size:])
                         if family == "surface_moments" or
                         family.startswith("top_face_") or
                         family.startswith("line_hybrid_") else
                         wrench(beam_interface,beam_nodes))
            work=float(load@full)
            physical_work=float(full[:shell_size]@physical@full[:shell_size])
            shell_total=float(full[:shell_size]@shell_k@full[:shell_size])
            balance=max(float(np.linalg.norm(residual[free])/max(1.,
                                      np.linalg.norm(load[free]))),
                        float(np.linalg.norm(constraint@full)),
                        float(abs(full@(primal@full)-work)),
                        float(np.linalg.norm(shell_wrench+beam_wrench)),
                        float(abs(interface@full)))
            if (not np.isfinite(full).all() or work<=0. or
                    physical_work<=0. or balance>1e-7):
                raise ArithmeticError(
                    f"graded terminal mechanics check failed: {family} "
                    f"{loading}, balance={balance:.6e}, "
                    f"shell_total={shell_total:.6e}, "
                    f"shell_physical={physical_work:.6e}, "
                    f"shell_nonphysical={shell_total-physical_work:.6e}")
            active=np.flatnonzero(np.max(np.abs(shell_interface),axis=1)>1e-10)
            local_kinematics={}
            interval_kinematics={}
            if has_local_stations and family.startswith("line_hybrid_"):
                nodal_shell=full[:shell_size].reshape(-1,6)
                for station in (900,950,1000):
                    x=station/1000
                    def shell_value_at(x_sample, y, axis):
                        nodes,shape=(structured_c1_mixed_point_shape(
                            xs,ys,x_sample,y,s3_cells,
                            diagonal=mixed_s3_diagonal) if mixed_s3 else
                            structured_q4_point_shape(xs,ys,x_sample,y))
                        return float(shape@nodal_shell[nodes,axis])
                    def shell_value(y, axis):
                        return shell_value_at(x,y,axis)
                    half_index=int(np.flatnonzero(
                        abs(beam.abscissae-x)<1e-12)[0])
                    def port_rx(half, weight):
                        element, cell=divmod(half,2)
                        port=NativeHalfCellSpatialPort(
                            beam.context,element,cell,weight)
                        return float((port.tangent(base)@
                                      full[shell_size:])[3])
                    footprint_ys=ys[(ys>=-.05-1e-12)&(ys<=.05+1e-12)]
                    if (len(footprint_ys)<2 or
                            abs(footprint_ys[0]+.05)>1e-12 or
                            abs(footprint_ys[-1]-.05)>1e-12):
                        raise ValueError("C1 shell top footprint edge incomplete")
                    edge_weights=np.zeros(len(footprint_ys))
                    half_intervals=.5*np.diff(footprint_ys)/.1
                    edge_weights[:-1]+=half_intervals
                    edge_weights[1:]+=half_intervals
                    shell_top_edge=np.array([
                        (shell_value(y,0)+.5*plate_thickness*shell_value(y,4),
                         shell_value(y,1)-.5*plate_thickness*shell_value(y,3),
                         shell_value(y,2)) for y in footprint_ys])
                    def patch_mean_at(x_sample, axis):
                        return float(sum(edge_weights[i]*shell_value_at(
                            x_sample,y,axis)
                            for i,y in enumerate(footprint_ys)))
                    local_kinematics[str(station)]={
                        "beam_nodal_translation_m":full[
                            shell_size+6*station_nodes[station]:
                            shell_size+6*station_nodes[station]+3].tolist(),
                        "beam_nodal_rotation_rad":full[
                            shell_size+6*station_nodes[station]+3:
                            shell_size+6*station_nodes[station]+6].tolist(),
                        "beam_nodal_rx_rad":float(full[
                            shell_size+6*station_nodes[station]+3]),
                        "beam_port_rx_left_rad":(
                            port_rx(half_index-1,1.)
                            if half_index>0 else None),
                        "beam_port_rx_right_rad":(
                            port_rx(half_index,0.)
                            if half_index<beam.n_half_cells else None),
                        "shell_top_slope_z_y_rad":(
                            shell_value(.05,2)-shell_value(-.05,2))/.1,
                        "shell_top_edge_sample_y_m":footprint_ys.tolist(),
                        "shell_top_edge_mean_translation_m":np.average(
                            shell_top_edge,axis=0,weights=edge_weights).tolist(),
                        "shell_director_rx_rad":shell_value(0.,3),
                        **({"shell_top_slope_z_x_rad":(
                            patch_mean_at(1.,2)-patch_mean_at(.96,2))/.04,
                            "shell_director_ry_rad":shell_value(0.,4),
                            "shell_local_patch_mean_ry_rad":(
                                patch_mean_at(.98,4))}
                           if station==1000 else {}),
                    }
                if family == "line_hybrid_two_intervals":
                    def shell_trace(x, y, axis):
                        nodes,shape=structured_q4_point_shape(xs,ys,x,y)
                        return float(shape@nodal_shell[nodes,axis])
                    for index,segment in enumerate(segments):
                        interval_kinematics[str(index)]={
                            "x_interval_m":list(segment.x_interval),
                            "beam_port_mean_rx_rad":float(sum(
                                point.weight*(point.port.tangent(base)@
                                              full[shell_size:])[3]
                                for point in segment.points)),
                            "shell_top_mean_slope_z_y_rad":float(sum(
                                point.weight*(
                                    shell_trace(point.x,.05,2)-
                                    shell_trace(point.x,-.05,2))/.1
                                for point in segment.points)),
                            "shell_director_mean_rx_rad":float(sum(
                                point.weight*shell_trace(point.x,0.,3)
                                for point in segment.points)),
                        }
            line_recovery={}
            if (line_gap_patch is not None and
                    family.startswith("line_hybrid_") and
                    not family.endswith("_without_pitch")):
                continuous = line_gap_patch.linearized_continuous_gap_statistics(
                    full[:shell_size], full[shell_size:])
                basis_work=[]
                for index in range(m//6):
                    row_slice=slice(6*index,6*(index+1))
                    local_action=-constraint[row_slice].T@multiplier[row_slice]
                    shell_part=local_action[:shell_size]
                    beam_part=local_action[shell_size:]
                    shell_work=float(shell_part@full[:shell_size])
                    beam_work=float(beam_part@full[shell_size:])
                    shell_resultant=wrench(
                        shell_part.reshape(-1,6),shell_nodes)
                    beam_resultant=retained_beam_wrench(beam_part)
                    if (abs(shell_work+beam_work)>1e-8 or
                            np.linalg.norm(shell_resultant+beam_resultant)>1e-8):
                        raise ArithmeticError(
                            "native line basis side work/wrench does not balance")
                    basis_work.append({
                        "basis_index":index,
                        "shell_work_j":shell_work,
                        "beam_work_j":beam_work,
                        "shell_wrench_N_Nm":shell_resultant.tolist(),
                        "beam_wrench_N_Nm":beam_resultant.tolist(),
                    })
                if abs(sum(row["shell_work_j"] for row in basis_work)-
                       float(interface[:shell_size]@full[:shell_size]))>1e-8:
                    raise ArithmeticError("native line basis work does not sum")
                channel_work=[]
                if include_tip_my and loading == "tip_my":
                    channel_action=np.zeros(total)
                    for channel,name in enumerate(("Fx","Fy","Fz",
                                                   "Mx","My","Mz")):
                        indices=np.arange(channel,m,6)
                        action=-(constraint[indices].T@multiplier[indices])
                        channel_action+=action
                        shell_part=action[:shell_size]
                        beam_part=action[shell_size:]
                        shell_work=float(shell_part@full[:shell_size])
                        beam_work=float(beam_part@full[shell_size:])
                        shell_resultant=wrench(
                            shell_part.reshape(-1,6),shell_nodes)
                        beam_resultant=retained_beam_wrench(beam_part)
                        if (abs(shell_work+beam_work)>1e-8 or
                                np.linalg.norm(shell_resultant+beam_resultant)>1e-8):
                            raise ArithmeticError(
                                "native line channel work/wrench does not balance")
                        channel_work.append({
                            "channel":name,"shell_work_j":shell_work,
                            "beam_work_j":beam_work,
                            "shell_wrench_N_Nm":shell_resultant.tolist(),
                            "beam_wrench_N_Nm":beam_resultant.tolist(),
                        })
                    if np.linalg.norm(channel_action-interface)>1e-8:
                        raise ArithmeticError("native line channel actions do not sum")
                line_recovery={
                    "continuous_reference_linearized_gap":continuous,
                    "basis_interface_side_work_and_wrench":basis_work,
                    **({"channel_interface_side_work_and_wrench":channel_work}
                       if channel_work else {}),
                    "meaning": ("physical -C.T lambda work on accepted "
                                "reference-linear KKT coordinates, including "
                                "native retained B3 half-cell variables; "
                                "not nonlinear accepted-state recovery"),
                }
            selected_recovery={}
            if recover_top_face_local:
                probes={}
                for probe_name, probe_pieces in top_face_patches.items():
                    sampled=[piece.linearized_gap_statistics(
                        full[:shell_size],full[shell_size:])
                        for piece in probe_pieces]
                    maximum=max(float(np.max(np.abs(
                        item["weak_six_gap_m_rad"]))) for item in sampled)
                    if (probe_name == family or
                            family == "top_face_eight_intervals") and maximum > 1e-8:
                        raise ArithmeticError("nested top-face weak gap did not close")
                    probes[probe_name]={
                        "per_interval":sampled,
                        "max_abs_weak_six_gap_m_rad":maximum,
                        "max_quadrature_position_gap_m":max(
                            item["quadrature_max_position_gap_m"]
                            for item in sampled),
                    }
                by_x=np.zeros(len(xs)-1)
                by_y=np.zeros(len(ys)-1)
                for i,j,dofs,local in shell_elements:
                    local_u=full[dofs]
                    local_work=float(local_u@local@local_u)
                    by_x[i]+=local_work
                    by_y[j]+=local_work
                if abs(float(by_x.sum())-physical_work)>1e-9:
                    raise ArithmeticError("selected shell work does not reassemble")
                intervals=[]
                for index,piece in enumerate(top_face_patches[family]):
                    row_slice=slice(6*index,6*(index+1))
                    local_action=-constraint[row_slice].T@multiplier[row_slice]
                    shell_work=float(local_action[:shell_size]@full[:shell_size])
                    beam_work=float(local_action[shell_size:]@full[shell_size:])
                    if abs(shell_work+beam_work)>1e-9:
                        raise ArithmeticError("interval interface work imbalance")
                    intervals.append({
                        "x_interval_m":list(piece.x_interval),
                        "basis_integral_m":piece.basis_integral_m,
                        "shell_work_j":shell_work,
                        "beam_work_j":beam_work,
                        "shell_wrench_N_Nm":wrench(
                            local_action[:shell_size].reshape(-1,6),
                            shell_nodes).tolist(),
                    })
                if abs(sum(item["shell_work_j"] for item in intervals)-
                       float(interface[:shell_size]@full[:shell_size]))>1e-9:
                    raise ArithmeticError("interval work does not sum")
                selected_recovery={
                    "gap_on_both_spaces":probes,
                    "physical_shell_work_by_x_cell_j":by_x.tolist(),
                    "physical_shell_work_by_y_cell_j":by_y.tolist(),
                    "shell_x_cell_intervals_m":[
                        [float(low),float(high)]
                        for low,high in zip(xs[:-1],xs[1:])],
                    "interface_interval_work_and_wrench":intervals,
                    "meaning":"u^T K_physical u work without one-half energy factor; gaps are sampled top-face minus native beam-bottom from the same solved state",
                }
            cases[loading]={
                "tip_rx_rad":float(full[shell_size+6*beam.n_half_cells+3]),
                "tip_ry_rad":float(full[shell_size+6*beam.n_half_cells+4]),
                "tip_uz_m":float(full[shell_size+6*beam.n_half_cells+2]),
                "beam_section_rx_at_stations_rad":{
                    str(station):float(full[shell_size+
                        6*station_nodes[station]+3])
                    for station in stations},
                **({"terminal_station_kinematics":local_kinematics}
                   if local_kinematics else {}),
                **({"interval_mean_kinematics":interval_kinematics}
                   if interval_kinematics else {}),
                **({"sampled_line_position_gap":{
                    key:value for key,value in
                    line_gap_patch.linearized_gap_statistics(
                        full[:shell_size],full[shell_size:]).items()
                    if key in ("quadrature_rms_position_gap_m",
                               "quadrature_max_position_gap_m",
                               "quadrature_points")}}
                   if line_gap_patch is not None and
                      family.startswith("line_hybrid_") else {}),
                **({"continuous_line_recovery":line_recovery}
                   if line_recovery else {}),
                **({"omitted_pitch_gap":
                    line_gap_patch.linearized_continuous_gap_statistics(
                        full[:shell_size],full[shell_size:])}
                   if family.endswith("_without_pitch") else {}),
                "total_external_work_j":work,
                "physical_shell_work_j":physical_work,
                "numerical_shell_work_j":shell_total-physical_work,
                "mechanics_error":balance,
                "shell_support_wrench_N_Nm":wrench(
                    residual[:shell_size].reshape(-1,6),shell_nodes).tolist(),
                "beam_root_wrench_N_Nm":wrench(
                    residual[shell_size:shell_size+beam.context.nodal_count]
                    .reshape(-1,6),beam_nodes).tolist(),
                "interface_shell_wrench_N_Nm":shell_wrench.tolist(),
                "interface_beam_wrench_N_Nm":beam_wrench.tolist(),
                "interface_shell_work_j":float(interface[:shell_size]@
                                               full[:shell_size]),
                **({"selected_top_face_local_recovery":selected_recovery}
                   if recover_top_face_local else {}),
                **({"constraint_multiplier_material_N_Nm":
                    multiplier.tolist()}
                   if family.startswith("line_hybrid_") else {}),
                "active_shell_nodes":[
                    {"node_row":int(i),"xyz_m":shell_nodes[i].tolist(),
                     "force_moment_N_Nm":shell_interface[i].tolist()}
                    for i in active],
                **({"surface_point_gap":surface_patch.linearized_gap_statistics(
                    full[:shell_size],full[shell_size:])}
                   if surface_patch is not None and family=="surface_moments"
                   else {}),
            }
        records[family]={"constraint_rank":rank,"cases":cases,
                         **({"scaled_row_inventory":row_inventory}
                            if row_inventory is not None else {}),
                         **({"basis_integrals_m":space_masses[family]}
                            if family in space_masses else {})}
    return {"patch_cells_per_side":(patch_cells if transverse_cells is None
                                     or transverse_cells==patch_cells else None),
            "bond_x_cells":patch_cells,
            "bond_y_cells":patch_cells if transverse_cells is None else transverse_cells,
            "plate_thickness_m":plate_thickness,
            "shell_nodes":len(shell_nodes),
            "e4_elements":(len(xs)-1)*(len(ys)-1)+len(s3_cells),
            "beam_half_cells":beam.n_half_cells,"results":records,
            **({"shell_topology":"Q4/S3-V2D"} if mixed_s3 else {}),
            **({"s3_diagonal":mixed_s3_diagonal} if mixed_s3 else {}),
            **({"surface_moment_reference":surface_diagnostics}
               if include_surface_moments else {}),
            **({"top_face_reference":top_face_reference}
               if top_face_counts else {})}


def run_graded_terminal(output_dir: Path,
                        patch_levels: tuple[int,...] = (4,8), *,
                        plate_thickness: float = .07,
                        include_surface_moments: bool = False,
                        c1_opening_shear: bool = False) -> None:
    if plate_thickness not in (.07,.035):
        raise ValueError("bounded native plate thickness required")
    if c1_opening_shear and (plate_thickness != .035 or
                             not include_surface_moments or
                             patch_levels != (2,4)):
        raise ValueError("matched C1 opening screen requires the two/four-cell surface patch")
    beam_axis_z=plate_thickness/2+.05
    source=subprocess.check_output(["git","rev-parse","HEAD"],
                                   text=True).strip()
    if subprocess.check_output(["git","status","--porcelain"],
                               text=True).strip():
        raise RuntimeError("run requires clean source")
    output_dir.mkdir(parents=True,exist_ok=False)
    manifest={
        "status":"EXPLORATORY_NOT_G0_ACCEPTANCE",
        "source_sha":source,
        "question":("On the exact C1 tangent-aligned beam reference, do the two bounded native terminal-patch grids reproduce the balanced torque and opening-shear work used in the matched-solid discriminator?"
                    if c1_opening_shear else
                    "Does a six-resultant physical surface-position bond with a distributed native beam trace improve torque/shear transfer while preserving work and rigid motions?"
                    if include_surface_moments else
                    "Does resolving the native coupled terminal patch change torque/shear work and interface-force distribution for rigid and distributing laws?"),
        "decision":("Reproduction with balanced work retains the surface-position footprint for independent mechanics review and a separately justified sparse patch study; disagreement requires correcting the native comparison. The saved solid torque case penetrates; no accuracy or true-line selection follows."
                    if c1_opening_shear else
                    "If reference, rank, work and physical response pass and the result settles from four to eight patch cells, advance finite variations and interface-stability assessment; failure or material drift requires revising the surface trace/section model. No law is production-selected by this screen."
                    if include_surface_moments else
                    "Settling work and transfer fields supports a matched 3D bond comparison; material drift requires revising patch trace or resolution policy first. Neither selects a production law or true-line multiplier."),
        "geometry":f"E4 Q4 plate [-.5,1.5]x[-.5,.5] m, t={plate_thickness:g} m; B3 centroid z={beam_axis_z:g} m; x=[.9,1], y=[-.05,.05] m patch",
        "plate_thickness_m":plate_thickness,
        "grids":("structured local Q4 with two/four cells per patch direction; exact C1 native half-cell cuts in surface quadrature"
                 if c1_opening_shear else
                 "structured local Q4 with four/eight cells per patch direction; exact area-mean/director/polar-yaw trace and corner rows"),
        "executed_patch_levels":list(patch_levels),
        "include_surface_moments":include_surface_moments,
        "beam":("native generalized B3, twelve nonuniform half-cells, exact 2 µm C1 tangent-aligned reference, square-section Saint-Venant torsion; root fixed"
                if c1_opening_shear else
                "native generalized B3, eight half-cells, square-section Saint-Venant torsion; root fixed"),
        "supports":"all E4 left-edge coordinates fixed and B3 root fixed",
        "loads":("separate +100 Nm tip Mx and -1000 N tip Fz opening"
                 if c1_opening_shear else
                 "separate +100 Nm tip Mx and +1000 N tip Fz"),
        **({"reference_profile":"c1_tangent_2um"} if c1_opening_shear else {}),
        "uniform_trace_parity":"structured Q4 exact trace and corner rows match the existing uniform native port at x8/y8 to <1e-12",
        "limits":{"max_dense_kkt_rows":1900,"blas_threads":1},
        "scope":"linearized bounded coupled diagnostic; no external accuracy, continuum interface traction, nonlinear or qualification claim",
        "python":platform.python_version(),"numpy":np.__version__,
    }
    with threadpool_limits(limits=1):
        manifest["effective_threadpools"]=threadpool_info()
        (output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        beam=build(12 if c1_opening_shear else 8,z_offset=beam_axis_z,
                   torsion_constant=square_torsion_constant(),
                   reference_profile=("c1_tangent_2um" if c1_opening_shear
                                      else "straight"))
        records=[]
        for patch_cells in patch_levels:
            started=time.perf_counter()
            result=graded_terminal_coupling(
                patch_cells,beam,plate_thickness=plate_thickness,
                include_surface_moments=include_surface_moments,
                force_sign=-1. if c1_opening_shear else 1.)
            records.append({"patch_cells_per_side":patch_cells,
                            "seconds":time.perf_counter()-started,
                            "response":result})
            (output_dir/"results.json").write_text(json.dumps(
                records,indent=2,allow_nan=False)+"\n")
            for law,law_result in result["results"].items():
                for loading,case in law_result["cases"].items():
                    print(json.dumps({"patch_cells":patch_cells,
                                      "law":law,"load":loading,
                                      "tip_rx_urad":1e6*case["tip_rx_rad"],
                                      "tip_z_um":1e6*case["tip_uz_m"],
                                      "active_shell_nodes":len(
                                          case["active_shell_nodes"])}),
                          flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--terminal-beam-resolution", action="store_true")
    parser.add_argument("--terminal-shell-resolution", action="store_true")
    parser.add_argument("--terminal-interface", action="store_true")
    parser.add_argument("--graded-terminal", action="store_true")
    parser.add_argument("--graded-patch-cells",type=int,choices=(4,8),
                        nargs="+",default=(4,8))
    parser.add_argument("--half-plate-thickness",action="store_true")
    parser.add_argument("--surface-moment-candidate",action="store_true")
    parser.add_argument("--c1-opening-shear",action="store_true")
    args = parser.parse_args()
    if sum((args.terminal_beam_resolution,args.terminal_shell_resolution,
            args.terminal_interface,args.graded_terminal))>1:
        parser.error("select one terminal resolution screen")
    if args.half_plate_thickness and not args.graded_terminal:
        parser.error("half plate thickness requires the graded terminal screen")
    if args.surface_moment_candidate and not args.graded_terminal:
        parser.error("surface moment candidate requires graded terminal screen")
    if args.c1_opening_shear and not args.graded_terminal:
        parser.error("C1 opening shear requires graded terminal screen")
    if args.graded_terminal:
        if args.c1_opening_shear and (args.half_plate_thickness or
                                      args.surface_moment_candidate or
                                      tuple(args.graded_patch_cells) != (4,8)):
            parser.error("C1 opening shear selects its own bounded model and grids")
        run_graded_terminal(args.output_dir,
                            (2,4) if args.c1_opening_shear else
                            tuple(args.graded_patch_cells),
                            plate_thickness=.035 if (args.half_plate_thickness or
                                                      args.c1_opening_shear) else .07,
                            include_surface_moments=(args.surface_moment_candidate or
                                                     args.c1_opening_shear),
                            c1_opening_shear=args.c1_opening_shear)
    elif args.terminal_interface:
        run_terminal_interface(args.output_dir)
    elif args.terminal_beam_resolution:
        run_terminal_beam_resolution(args.output_dir)
    elif args.terminal_shell_resolution:
        run_terminal_shell_resolution(args.output_dir)
    else:
        run(args.output_dir)
