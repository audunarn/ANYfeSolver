"""Focused regression tests for the adopted native S3 replay roundoff bound.

Root adoption (see .restart-ci/ROOT_STATE_REPLAY_DECISION.md): a binary64
same-predecessor replay of an accepted S3 update may round plastic-strain
entries at the last ulp without changing the material path. The saved
accepted state and its serialized bytes stay authoritative; every other
sealed field must stay exact, and the plastic arrays may differ only within
the existing 64-eps array-scale relative roundoff bound. Rehashed states that
change any other field, or plastic beyond that bound, are still refused.
"""
from pathlib import Path

import numpy as np
import pytest

from anyfesolver._stiffened_shell_ge_finite_experimental import _require_native_s3_replay
from anyfesolver.e4_pl_s3_v2d_state import seal_v2d_state
from test_stiffened_shell_native_point_displacement import owner
from test_stiffened_shell_native_plastic_s3_displacement import definition

FIXTURE = (Path(__file__).parent /
           "data/native_mixed_j2_displacement/adaptive-midpoint-v1.bin")


def restored(request):
    model = definition(True, True)
    analysis = owner(request, model)
    raw = FIXTURE.read_bytes()
    prefix = analysis.restore(raw)
    return analysis, prefix, raw


def replay_inputs(analysis):
    driver = analysis._finite_driver
    trial = driver._state
    accepted = driver._shell_origins
    q4_count = len(analysis._definition.shell.q4_elements)
    candidates = list(accepted[:q4_count])
    for state in accepted[q4_count:]:
        candidates.append(seal_v2d_state(
            {**state, "solver_kinematics": "ELEMENT_LOCAL"}))
    return trial, accepted, q4_count, candidates


def flowed_entry(accepted, q4_count):
    best = None
    for index, state in enumerate(accepted[q4_count:]):
        plastic = np.asarray(state["plastic_strain"], dtype=float)
        flat = int(np.argmax(np.abs(plastic)))
        row, column = divmod(flat, plastic.shape[1])
        if best is None or abs(plastic[row, column]) > abs(best[0]):
            best = (plastic[row, column], q4_count + index, row, column)
    assert best is not None and best[0] != 0., "fixture S3 plastic flow required"
    return best


def resealed(accepted, q4_count, index, name, value):
    made = {**accepted[index]}
    array = np.asarray(made[name], dtype=float).copy()
    row, column = value[2], value[3]
    array[row, column] = value[0]
    made[name] = array
    return accepted[:index] + (seal_v2d_state(made),) + accepted[index + 1:]


def test_midpoint_fixture_restores_and_roundtrips_byte_for_byte(request):
    analysis, prefix, raw = restored(request)
    assert all(np.max(x.alpha) > 0. for x in prefix.shell_s3_material)
    assert analysis.checkpoint() == raw
    trial, accepted, q4_count, candidates = replay_inputs(analysis)
    _require_native_s3_replay(analysis, trial, accepted, candidates)


def test_s3_replay_accepts_only_bounded_plastic_rounding(request):
    analysis, _, _ = restored(request)
    trial, accepted, q4_count, candidates = replay_inputs(analysis)
    value = flowed_entry(accepted, q4_count)
    index = value[1]
    spacing = np.spacing(np.abs(value[0]))
    # Last-ulp plastic rounding alone is the adopted binary64 replay difference.
    rounded = resealed(accepted, q4_count, index, "plastic_strain",
                       (value[0] + spacing, index, value[2], value[3]))
    _require_native_s3_replay(analysis, trial, rounded, candidates)
    # Plastic differences beyond the 64-eps array-scale bound are refused.
    gross = resealed(accepted, q4_count, index, "plastic_strain",
                    (value[0] * (1. + 1e-9), index, value[2], value[3]))
    with pytest.raises(ValueError,
                       match="native S3 checkpoint response disagrees"):
        _require_native_s3_replay(analysis, trial, gross, candidates)
    # Any rehashed non-plastic field is refused, even at the last ulp.
    alpha = np.asarray(accepted[index]["alpha"], dtype=float)
    alpha_row = int(np.argmax(alpha))
    made = {**accepted[index]}
    alpha = alpha.copy()
    alpha[alpha_row] = np.nextafter(alpha[alpha_row], np.inf)
    made["alpha"] = alpha
    rehashed = (accepted[:index] + (seal_v2d_state(made),) +
                accepted[index + 1:])
    with pytest.raises(ValueError,
                       match="native S3 checkpoint response disagrees"):
        _require_native_s3_replay(analysis, trial, rehashed, candidates)
    # Rehashed accepted coordinates are refused even with consistent plastic.
    coordinates = np.asarray(
        accepted[index]["committed_constitutive_u"], dtype=float).copy()
    coordinates[0] = np.nextafter(coordinates[0], np.inf)
    made = {**accepted[index], "committed_constitutive_u": coordinates}
    rehashed = (accepted[:index] + (seal_v2d_state(made),) +
                accepted[index + 1:])
    with pytest.raises(ValueError,
                       match="native S3 checkpoint response disagrees"):
        _require_native_s3_replay(analysis, trial, rehashed, candidates)
