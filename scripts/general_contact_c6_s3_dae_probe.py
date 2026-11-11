"""Development diagnostic for differentiated S3 algebraic velocity work."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_general_contact_transient_plane import _guided_s3_descriptor_impact

from anyfesolver import ContactTransientSolverControls, TransientConfig, solve_transient_contact
from anyfesolver.algebraic_dynamics import build_declared_algebraic_basis
from anyfesolver.assembly import assemble_mass_matrix, build_constraint_transformation
from anyfesolver.general_contact.core import contact_scope
from anyfesolver.matrix_assembly import assemble_stiffness_matrix
from anyfesolver.nonlinear_static import _assemble_nonlinear_system


def run(dt: float) -> dict:
    model, contact, velocity = _guided_s3_descriptor_impact(tilt=.002, span=.1)
    result = solve_transient_contact(
        model, contact, TransientConfig(dt, .012, initial_velocity=velocity),
        solver_controls=ContactTransientSolverControls(
            allow_s3_descriptor_dev=True,
        ),
    )
    K0, _ = assemble_stiffness_matrix(model)
    M, _ = assemble_mass_matrix(model)
    _, _, T, _, independent, _ = build_constraint_transformation(
        K0, np.zeros(M.shape[0]), model,
    )
    Z = build_declared_algebraic_basis(
        model, M, (T.T @ M @ T).tocsr(), T, independent,
        dense_size_limit=512,
    ).reduced_basis
    null_position_defects = []
    null_velocity_defects = []
    full_position_defects = []
    full_velocity_defects = []
    stage_defects = []
    for index in range(1, len(result.times)):
        step = float(result.times[index]-result.times[index-1])
        previous_position = result.displacements[index-1, independent]
        current_position = result.displacements[index, independent]
        previous_velocity = result.velocities[index-1, independent]
        current_velocity = result.velocities[index, independent]
        previous_acceleration = result.accelerations[index-1, independent]
        current_acceleration = result.accelerations[index, independent]
        position_defect = (
            current_position-previous_position-step*previous_velocity
            -step*step*(0.25*previous_acceleration+0.25*current_acceleration)
        )
        velocity_defect = (
            current_velocity-previous_velocity
            -0.5*step*(previous_acceleration+current_acceleration)
        )
        full_position_defects.append(float(np.linalg.norm(position_defect)))
        full_velocity_defects.append(float(np.linalg.norm(velocity_defect)))
        null_position_defects.append(
            float(np.linalg.norm(Z.T @ position_defect))
        )
        null_velocity_defects.append(
            float(np.linalg.norm(Z.T @ velocity_defect))
        )
        stage_defects.append({
            "end_time_s": float(result.times[index]),
            "active_before": result.contact[index-1].active_pressure_dofs,
            "active_after": result.contact[index].active_pressure_dofs,
        })
    with contact_scope(model, contact,
                       initial_displacement=np.zeros(M.shape[0])):
        full = result.displacements[-1]
        _, tangent, _ = _assemble_nonlinear_system(
            model, full, {}, 3, tangent=True, kinematics="von_karman",
        )
        independent_step_s = 5.0e-6
        step_direction = independent_step_s*result.velocities[-1]
        _, tangent_plus, _ = _assemble_nonlinear_system(
            model, full+step_direction, {}, 3, tangent=True,
            kinematics="von_karman",
        )
        _, tangent_minus, _ = _assemble_nonlinear_system(
            model, full-step_direction, {}, 3, tangent=True,
            kinematics="von_karman",
        )
    reduced_tangent = (T.T @ tangent @ T).tocsr()
    reduced_velocity = result.velocities[-1, independent]
    tangent_velocity = np.asarray(
        reduced_tangent @ reduced_velocity,
    ).reshape(-1)
    algebraic_rate = np.asarray(Z.T @ tangent_velocity).reshape(-1)
    tangent_acceleration = np.asarray(
        reduced_tangent @ result.accelerations[-1, independent]
    ).reshape(-1)
    hessian_velocity = np.asarray(
        T.T @ ((tangent_plus-tangent_minus) @ result.velocities[-1])
    ).reshape(-1)/(2.0*independent_step_s)
    acceleration_defect = np.asarray(
        Z.T @ (tangent_acceleration+hessian_velocity)
    ).reshape(-1)
    return {
        "dt_s": dt,
        "final_time_s": float(result.times[-1]),
        "final_minimum_gap_m": float(result.contact[-1].minimum_gap),
        "active_pressure_nodes": result.contact[-1].active_pressure_dofs,
        "maximum_full_newmark_position_defect_m": max(full_position_defects),
        "maximum_full_newmark_velocity_defect_m_per_s": max(full_velocity_defects),
        "maximum_null_newmark_position_defect_m": max(null_position_defects),
        "maximum_null_newmark_velocity_defect_m_per_s": max(null_velocity_defects),
        "maximum_null_position_defect_stage": stage_defects[int(np.argmax(null_position_defects))],
        "maximum_null_velocity_defect_stage": stage_defects[int(np.argmax(null_velocity_defects))],
        "maximum_unchanged_activity_null_velocity_defect_m_per_s": max(
            (defect for defect, stage in zip(null_velocity_defects, stage_defects)
             if stage["active_before"] == stage["active_after"]), default=0.0,
        ),
        "algebraic_velocity_defect_n_per_s": float(np.linalg.norm(algebraic_rate)),
        "full_force_rate_scale_n_per_s": float(np.linalg.norm(tangent_velocity)),
        "relative_algebraic_velocity_defect": float(
            np.linalg.norm(algebraic_rate)
            / max(1.0e-30, np.linalg.norm(tangent_velocity))
        ),
        "algebraic_acceleration_defect_n_per_s2": float(
            np.linalg.norm(acceleration_defect)
        ),
        "full_acceleration_rate_scale_n_per_s2": float(
            np.linalg.norm(tangent_acceleration+hessian_velocity)
        ),
    }


if __name__ == "__main__":
    for step in (.0002, .0001, .00005):
        print(run(step))
