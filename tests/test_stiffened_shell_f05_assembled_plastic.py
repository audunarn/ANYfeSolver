"""F05 composition checks; no historical authority-bound probe is replayed."""

from dataclasses import replace
from hashlib import sha256
import json

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from anyfesolver import stiffened_shell
from anyfesolver._stiffened_shell_ge_finite_experimental import _material_arrays
from anyfesolver.e4_pl_s3_state import canonical_json_bytes
from test_stiffened_shell_ge_finite_experimental import _actions
from test_stiffened_shell_q4_pullback_owner import _definition


@pytest.fixture(scope="module")
def selected_mixed_predecessor():
    definition = _definition(plastic=True, mixed=True)
    owner = stiffened_shell.create_analysis(definition=definition)
    owner.solve(_actions(), (.5, 1.), max_iterations=20, max_backtracks=12)
    return definition, owner.checkpoint()


@pytest.mark.parametrize("anchored", [False, True])
def test_active_q4_s3_and_native_mixed_actual_update_derivative(
        selected_mixed_predecessor, anchored, tmp_path):
    definition, checkpoint = selected_mixed_predecessor
    owner = stiffened_shell.create_analysis(definition=definition)
    owner.restore(checkpoint)
    if anchored:
        owner.rebase_rotations(owner.recover().generation)
    saved = owner.checkpoint()
    driver = owner._finite_driver
    origins = driver._shell_origins
    before = tuple(canonical_json_bytes(origin) for origin in origins)
    report = {"anchored": anchored, "load_factor": 2., "candidate_amount": .5,
              "axial_trial_strain": 2e-5, "steps": (.01, .001), "directions": [],
              "accepted_before_sha256": sha256(saved).hexdigest()}
    try:
        args = (2., origins, driver._beam_origins)
        residual, tangent, *_ = driver._assemble(driver._state, *args)
        count = owner._K.shape[0]
        free = np.r_[owner._free, np.arange(count, tangent.shape[0])]
        step = np.zeros(tangent.shape[0])
        step[free] = spsolve(tangent[free, :][:, free], -residual[free])
        candidate = driver._advance(driver._state, step, .5)
        # The bending-only loaded fixture leaves the outer S3 cells elastic.
        # A small prescribed axial trial activates both formulations from the
        # same accepted predecessor. Root supports remain satisfied; attachment
        # gaps need not vanish in this unaccepted derivative trial.
        axial = np.zeros_like(step)
        axial[:owner._shell_count].reshape(-1, 6)[:, 0] = (
            report["axial_trial_strain"]*owner._shell_xyz[:, 0])
        np.testing.assert_array_equal(axial[owner._fixed], 0.)
        candidate = driver._advance(candidate, axial, 1.)
        base = driver._assemble(candidate, *args)
        report["matrix_rows"] = base[1].shape[0]
        report["matrix_nnz"] = base[1].nnz
        report["reference_coordinate_scales"] = owner._coordinate_scale.tolist()
        masks = tuple(np.asarray(state["alpha"]) > np.asarray(origin["alpha"])+1e-12
                      for state, origin in zip(base[4], origins))
        report["shell_active_counts"] = [int(mask.sum()) for mask in masks]
        # This is specifically the selected full Q4 and actively updating S3
        # composition, not an elastic-shell or beam-only plastic derivative.
        assert all(mask.any() for mask in masks), report
        beam_origin_alpha = _material_arrays(driver._beam_origins)[1]
        beam_mask = _material_arrays(base[5])[1] > beam_origin_alpha+1e-12
        report["beam_active_count"] = int(beam_mask.sum())
        assert beam_mask.any()
        report["multiplier_norm"] = float(np.linalg.norm(candidate.multipliers))
        assert report["multiplier_norm"] > 1.
        no_multipliers = replace(candidate, multipliers=np.zeros_like(candidate.multipliers))
        zero_tangent = driver._assemble(no_multipliers, *args)[1]
        curvature = base[1][:count, :count]-zero_tangent[:count, :count]
        report["multiplier_curvature_norm"] = float(np.linalg.norm(curvature.data))
        assert report["multiplier_curvature_norm"] > 1e-8
        for direction in (step, step*np.cos(.3+np.arange(len(step)))+.1*axial):
            predicted = base[1]@direction
            errors = []
            for h in report["steps"]:
                plus = driver._assemble(driver._advance(candidate, direction, h), *args)
                minus = driver._assemble(driver._advance(candidate, direction, -h), *args)
                for evaluation in (plus, minus):
                    for state, origin, mask in zip(evaluation[4], origins, masks):
                        np.testing.assert_array_equal(np.asarray(state["alpha"]) >
                                                      np.asarray(origin["alpha"])+1e-12, mask)
                    np.testing.assert_array_equal(
                        _material_arrays(evaluation[5])[1] > beam_origin_alpha+1e-12,
                        beam_mask)
                observed = (plus[0]-minus[0])/(2*h)
                error = observed-predicted
                row_errors = {}
                for name, slots in (("shell", np.arange(owner._shell_count)),
                                    ("beam", np.arange(owner._shell_count, count)),
                                    ("free_primal", owner._free)):
                    denominator = np.linalg.norm(predicted[slots])
                    assert denominator > 1.
                    raw = float(np.linalg.norm(error[slots])/denominator)
                    weights = owner._coordinate_scale[slots]
                    weighted = float(np.linalg.norm(error[slots]*weights)/
                                     np.linalg.norm(predicted[slots]*weights))
                    row_errors[name] = {"relative": raw, "work_scaled_relative": weighted}
                row_errors["constraints_scaled_max"] = float(np.max(
                    np.abs(error[count:])/owner._constraint_scale, initial=0.))
                errors.append(row_errors)
                for name in ("shell", "beam", "free_primal"):
                    assert row_errors[name]["relative"] <= 1e-5, row_errors
                    assert row_errors[name]["work_scaled_relative"] <= 1e-5, row_errors
                assert row_errors["constraints_scaled_max"] <= 1e-9, row_errors
            report["directions"].append(errors)
        assert owner.checkpoint() == saved
        assert tuple(canonical_json_bytes(origin) for origin in origins) == before
        report["accepted_bytes_unchanged"] = True
        report["predecessors_unchanged"] = True
        report["sampled_shell_and_beam_masks_stable"] = True
        report["accepted_after_sha256"] = sha256(owner.checkpoint()).hexdigest()
    finally:
        (tmp_path/"f05-assembled.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        owner.close()
