"""Read-only paired pressure response; no contact admission or stability cutoff.

Spatial nodal force work and additive rotation-chart gap derivatives are kept
distinct. Singular values describe the explicitly supplied norms and pose.
They do not establish coercivity or mesh-uniform pressure stability.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PressureResponseScales:
    force_n: float
    moment_nm: float
    length_m: float
    reference_area_m2: float
    angle_rad: float = 1.0

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0.0 for v in
               (self.force_n, self.moment_nm, self.length_m,
                self.reference_area_m2, self.angle_rad)):
            raise ValueError("pressure diagnostic scales must be finite positive SI values")


def _spectrum(matrix):
    matrix = np.asarray(matrix, dtype=float)
    _, values, vectors = np.linalg.svd(matrix, full_matrices=True)
    resolution = np.finfo(float).eps * max(matrix.shape) * (values[0] if len(values) else 0.)
    rank = int(np.count_nonzero(values > resolution))
    domain = matrix.shape[1]
    return dict(singular_values=values, numerical_resolution=resolution,
                domain_dimension=domain, numerically_resolved_nullity=domain-rank,
                numerically_resolved_rank=rank,
                condition_number=float(values[0]/values[-1]) if len(values) and rank == domain else float("inf"),
                weakest_right_vector=vectors[-1] if len(vectors) else np.empty(0),
                resolution_scope="floating point SVD resolution only; no physical cutoff")


def paired_pressure_response(tangent, gap_chart, force_spin, support, moment_rows,
                             *, scales, active_modes=None):
    """Solve K D = B.T and report G D with all supported signed modes retained.

    Inputs already use independent unconstrained coordinates. ``force_spin``
    rows are spatial force/moment work, ``gap_chart`` rows are chart derivatives.
    Support is the current positive reference-measure overlap integral of Q1.
    The pressure normalization is the declared lumped norm sqrt(sum S lambda^2).
    """
    k = np.asarray(tangent.toarray() if hasattr(tangent, "toarray") else tangent, dtype=float)
    g = np.asarray(gap_chart, dtype=float)
    b = np.asarray(force_spin, dtype=float)
    s = np.asarray(support, dtype=float)
    moments = np.asarray(moment_rows, dtype=bool)
    if (not isinstance(scales, PressureResponseScales) or k.ndim != 2
            or k.shape[0] != k.shape[1] or g.ndim != 2 or b.shape != g.shape
            or g.shape[1] != k.shape[0] or s.shape != (g.shape[0],)
            or moments.shape != (k.shape[0],) or not len(s)
            or np.any(s <= 0.) or any(not np.all(np.isfinite(a)) for a in (k, g, b, s))):
        raise ValueError("paired diagnostic requires finite compatible matrices and every positive support")
    active = np.ones(len(s), dtype=bool) if active_modes is None else np.asarray(active_modes, dtype=bool)
    if active.shape != s.shape:
        raise ValueError("active pressure mask must preserve the complete supported space")
    dz_scale = scales.force_n*np.sqrt(s/scales.reference_area_m2)
    residual_scale = np.where(moments, scales.moment_nm, scales.force_n)
    coordinate_scale = np.where(moments, scales.angle_rad, scales.length_m)
    displacement = np.linalg.solve(k, b.T)
    response = g @ displacement
    force_scaled = b.T*dz_scale[None, :]/residual_scale[:, None]
    displacement_scaled = displacement*dz_scale[None, :]/coordinate_scale[:, None]
    gap_scaled = scales.force_n/(scales.length_m*scales.reference_area_m2)*np.sqrt(s)[:, None]*response*np.sqrt(s)[None, :]
    k_scaled = k*coordinate_scale[None, :]/residual_scale[:, None]
    backward = k @ displacement - b.T
    denominator = np.linalg.norm(k)*np.linalg.norm(displacement)+np.linalg.norm(b)
    scaled_backward = k_scaled @ displacement_scaled - force_scaled
    scaled_denominator = np.linalg.norm(k_scaled)*np.linalg.norm(displacement_scaled)+np.linalg.norm(force_scaled)
    return dict(schema="PAIRED_SPATIAL_SPIN_CHART_PRESSURE_RESPONSE_DEV_V0",
                support=s.copy(), active_mode_indices=np.flatnonzero(active),
                pressure_force_scale=dz_scale, displacement_per_force=displacement,
                gap_per_force=response, force_detection=_spectrum(force_scaled),
                displacement_response=_spectrum(displacement_scaled),
                gap_response=_spectrum(gap_scaled),
                active_gap_response=_spectrum(gap_scaled[np.ix_(active, active)]) if np.any(active) else None,
                scaled_tangent=_spectrum(k_scaled),
                solve_backward_error=float(np.linalg.norm(backward)/denominator) if denominator else 0.,
                dimensionless_solve_backward_error=float(np.linalg.norm(scaled_backward)/scaled_denominator) if scaled_denominator else 0.,
                gap_response_relative_asymmetry=float(np.linalg.norm(gap_scaled-gap_scaled.T)/np.linalg.norm(gap_scaled)) if np.any(gap_scaled) else 0.,
                tangent_relative_asymmetry=float(np.linalg.norm(k_scaled-k_scaled.T)/np.linalg.norm(k_scaled)) if np.any(k_scaled) else 0.,
                chart_spin_relative_difference=float(np.linalg.norm(g-b)/max(np.linalg.norm(g), np.linalg.norm(b))) if np.any(g) or np.any(b) else 0.,
                physical_admission=False, engineering_acceptance=False,
                norm_convention="current-support lumped Q1 pressure; separate spatial force/moment and chart coordinate scales")


def finite_pressure_pose(model, manager, displacement, nodal_forces, free_dofs,
                         *, program, scales, committed_states=None):
    """Consume the current finite contributor and actual structural tangent.

    A saved pose is a fixture, not a restart. Supplied states must already have
    the owner's validated local coordinate interpretation; an empty mapping
    means elastic-origin re-evaluation, never a history reconstruction.
    """
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system, _discard_nonlinear_state_candidate
    from .core import suspend_contact
    from .s3_patch_static import FINITE_Q4_Q4_MONOLITHIC_PROFILE
    from .semismooth import linearize, complementarity, reference_nodal_support
    if manager.profile != FINITE_Q4_Q4_MONOLITHIC_PROFILE or manager.finite_skin is None:
        raise ValueError("paired finite pressure diagnostic requires the exact finite Q4 profile")
    q = np.asarray(displacement, dtype=float)
    free = np.asarray(free_dofs, dtype=int)
    if (q.shape != (manager.n_dof,) or free.ndim != 1 or not len(free)
            or len(np.unique(free)) != len(free) or np.any(free < 0) or np.any(free >= len(q))):
        raise ValueError("diagnostic coordinates must be distinct independent model DOFs")
    if any(sum(int(d) in free for d in ds[3:]) not in (0, 3) for ds in manager.finite_skin.dofs):
        raise ValueError("finite pressure diagnostic requires complete angular triplets")
    names = {name: i for i, name in enumerate(('ux', 'uy', 'uz', 'rx', 'ry', 'rz'))}
    fixed = set()
    if model.constraint_equations:
        raise ValueError("pressure diagnostic supports literal selectors, not general MPC rows")
    for condition in model.boundary_conditions:
        for node in condition.node_ids:
            for name in condition.dof_constraints:
                fixed.add(model.mesh.dof_manager.get_node_dofs(node)[names[name]])
    for control in program.controls:
        if len(control.terms) != 1 or control.terms[0][2] != 1.:
            raise ValueError("pressure diagnostic requires exact nodal selectors")
        node, name, _ = control.terms[0]
        fixed.add(model.mesh.dof_manager.get_node_dofs(node)[names[name]])
    expected = sorted(set(range(manager.n_dof))-fixed)
    if free.tolist() != expected:
        raise ValueError("pressure diagnostic P must be the literal static and path selector complement")
    manager.require_finite_deformation(q)
    trial = None
    try:
        with suspend_contact():
            internal, structural, trial = _assemble_nonlinear_system(
                model, q, {} if committed_states is None else committed_states,
                5, tangent=True, kinematics="corotational",
                corotational_tangent="consistent", require_full_coordinates=True)
        contact = linearize(manager, q, nodal_forces, tangent=True)
        if contact.force_gradient is None:
            raise ValueError("finite diagnostic lacks separate spatial-spin pressure work")
        _, active, _, _ = complementarity(contact.state, nodal_forces,
            manager.stiffness*reference_nodal_support(manager.law), contact.gap_gradient, tangent=False)
        k = (structural+contact.tangent)[free][:, free]
        g = contact.gap_gradient[:, free]; b = contact.force_gradient[:, free]
        rotational = set(int(d) for ds in manager.finite_skin.dofs for d in ds[3:])
        report = paired_pressure_response(k, g, b, contact.state.reference_support,
                    [d in rotational for d in free], scales=scales, active_modes=active)
        report.update(pressure_mode_ids=tuple(manager.law.pressure_mode_ids),
                      free_dofs=free.copy(), structural_state_scope="validated supplied state or explicitly elastic origin")
        report['selector_provenance'] = dict(method="literal_static_bc_and_path_unit_selector_complement", constrained_dofs=sorted(fixed))
        return report, contact, structural, np.asarray(internal), k, g, b
    finally:
        _discard_nonlinear_state_candidate(trial)
