"""One bounded coupled static probe with analytic Q4/S3 pressure response."""

from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))

from anyfesolver import ContactPathControl, solve_static_nonlinear
from test_general_contact_supported_schur_path import _case


if __name__ == "__main__":
    model, definition, settings, program = _case()
    maximum_schur_attempts = 1
    maximum_relinearizations = 0
    if len(sys.argv) == 2 or (len(sys.argv) in (3, 4) and all(
            value.isdigit() for value in sys.argv[1:])):
        normal_control_node = int(sys.argv[1])
        if len(sys.argv) >= 3:
            maximum_schur_attempts = int(sys.argv[2])
        if len(sys.argv) == 4:
            maximum_relinearizations = int(sys.argv[3])
        program = replace(program, controls=(
            program.controls[0],
            ContactPathControl.node(normal_control_node, "uz"),
        ))
        print(f"normal_control_node={normal_control_node}")
    if len(sys.argv) == 3 and not all(value.isdigit() for value in sys.argv[1:]):
        shift_x, shift_y = map(float, sys.argv[1:])
        for node in (7, 8, 9, 10):
            model.mesh.get_node(node).x += shift_x
            model.mesh.get_node(node).y += shift_y
        print(f"additional_target_shift_m=({shift_x}, {shift_y})")
    settings = replace(
        settings,
        general_s3_differentiation="jet2_q4_s3_dev",
        q4_source_support_gradient_tolerance=1e-5,
        q4_source_support_hessian_tolerance=1e-5,
        q4_source_weighted_gap_gradient_tolerance=1e-5,
        q4_source_weighted_gap_hessian_tolerance=1e-5,
        maximum_q4_source_integration_triangles=128,
        augmentation_schur_maximum_attempts=maximum_schur_attempts,
        augmentation_schur_maximum_relinearizations=maximum_relinearizations,
    )
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=12,
        emit_restart_checkpoint=True,
    )
    print(f"status={result.status} reason={result.failure_reason} "
          f"accepted_steps={len(result.steps)}")
    failure = result.info.get("contact_failure", {})
    trace = failure.get("augmentation_trace", ())
    accepted_record = None
    if not trace and result.steps:
        accepted_record = result.info["force_displacement_history"][-1]
        trace = accepted_record.get("augmentation_trace", ())
    for index, entry in enumerate(trace, 1):
        print(f"augmentation={index} "
              f"pressure_change_pa={entry.get('pressure_change')} "
              f"schur_proposed={entry.get('schur_proposed')} "
              f"proposal_kind={entry.get('schur_proposal_kind')} "
              f"schur_ineligible={entry.get('schur_ineligible')} "
              f"schur_condition={entry.get('schur_scaled_condition')} "
              f"schur_singular_values={entry.get('schur_scaled_singular_values')} "
              f"weak_rhs_fraction={entry.get('schur_scaled_weak_rhs_fraction')} "
              f"weak_right_mode={entry.get('schur_scaled_weak_right_mode')} "
              f"pressure_gram_rank={entry.get('pressure_gram_scaled_rank')} "
              f"pressure_gram_condition={entry.get('pressure_gram_scaled_condition_estimate')} "
              f"force_map_singular_values={entry.get('schur_force_map_scaled_singular_values')} "
              f"unconstrained_force_singular_values={entry.get('schur_force_map_unconstrained_singular_values')} "
              f"single_control_singular_values={entry.get('schur_force_map_single_control_singular_values')} "
              f"schur_applied={entry.get('schur_applied')} "
              f"schur_rejected={entry.get('schur_rejected')} "
              f"candidate_evaluation_error={entry.get('candidate_evaluation_error')} "
              f"schur_relinearization={entry.get('schur_relinearization')} "
              f"schur_relinearization_proposed={entry.get('schur_relinearization_proposed')} "
              f"schur_actual_to_prior_ratio={entry.get('schur_actual_to_prior_ratio')} "
              f"schur_predicted_active_indices={entry.get('schur_predicted_active_indices')} "
              f"schur_actual_active_indices={entry.get('schur_actual_active_indices')} "
              f"schur_gap_tolerance_met={entry.get('schur_gap_tolerance_met')} "
              f"predicted_before={entry.get('schur_predicted_residual_before_pa')} "
              f"predicted_after={entry.get('schur_predicted_residual_pa')} "
              f"step_fraction={entry.get('schur_step_fraction')} "
              f"full_step_max_pa={entry.get('schur_full_step_max_pa')} "
              f"initial_pressure_pa={entry.get('schur_initial_pressure_pa')} "
              f"full_direction_pa={entry.get('schur_full_direction_pa')} "
              f"best_fixed_branch_fraction={entry.get('schur_best_fixed_branch_fraction')} "
              f"best_fixed_branch_residual_pa={entry.get('schur_best_fixed_branch_residual_pa')} "
              f"backtrack_rejections={entry.get('schur_backtrack_rejections')}")
    if failure:
        print(f"gap_tolerance_met={failure.get('gap_tolerance_met')} "
              f"active_modes={failure.get('active_pressure_dofs')} "
              f"schur_attempts={failure.get('schur_attempts')} "
              f"schur_relinearizations={failure.get('schur_relinearizations')} "
              f"schur_applied={failure.get('schur_applied')}")
    if accepted_record:
        print(f"accepted_pressure_change_pa={accepted_record.get('normal_multiplier_change')} "
              f"schur_attempts={accepted_record.get('schur_attempts')} "
              f"schur_relinearizations={accepted_record.get('schur_relinearizations')} "
              f"schur_applied={accepted_record.get('schur_applied')} "
              f"checkpoint={result.restart_checkpoint is not None}")
