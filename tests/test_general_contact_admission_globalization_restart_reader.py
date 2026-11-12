"""Pure-reader tamper cases for the enabled continuation restart trace.

Connected source review repair (v9): the independent Mistral review
found that the enabled restart trace reader accepted chronologies the
producer can never emit at the sequence level even though every span
was individually reachable — an accepted span can grow by at most the
single doubling the producer performs on acceptance (a 2^-6 span can
never be followed by a 0.5 span without the intermediate accepted
substeps), the recorded halvings_used must satisfy the producer
identity 1 + trial_count - accepted_substep_count, the failures implied
by the spans (including producer-permitted repeated failures on the
final truncated interval) must fit that count, the
external-nodal-work key must be present exactly when the program
declares nodal forces, and the recorded stagnation controls must equal
the live settings (not merely sit in valid ranges). These checks
exercise the tightened reader directly on synthetic records: no solve,
no fixture, no kernel oracle. Compatible producer-reachable records
(including a non-power-of-two final span and a repeated final-interval
failure) must still pass; every tampered shape must fail closed. The
expensive real-checkpoint compatibility (clean restart completing the
trajectory) stays in the driver module.
"""
import numpy as np
import pytest

from anyfesolver.general_contact.static_path import (
    _CONTACT_PATH_PHYSICAL_SUBSTEPS_REVISION,
    _validate_continuation_history_record,
)


class _StubProgram:
    """Minimal program surface consumed by the pure reader."""

    def __init__(self, total_steps=2, controls=2, nodal_forces=(0,)):
        self.total_steps = total_steps
        self.controls = tuple(range(controls))
        self.nodal_forces = tuple(nodal_forces)

    def target_at(self, step):
        return np.asarray([0.5 * step, -0.25 * step], dtype=float)


class _StubSettings:
    """Minimal live settings surface consumed by the pure reader."""

    def __init__(self, maximum_halvings=6, patience=2,
                 fraction=0.125, merit_rate=0.05):
        self.path_continuation_maximum_halvings = maximum_halvings
        self.path_stagnation_patience = patience
        self.path_stagnation_fraction = fraction
        self.path_stagnation_merit_rate = merit_rate


def _substep(start, end, **overrides):
    record = {
        "tau_start": float(start),
        "tau_end": float(end),
        "iterations": 3,
        "residual_norm": 1.0e-8,
        "constraint_error": 1.0e-9,
        "complementarity_relative_force": 1.0e-7,
        "candidate_evaluations": 5,
        "backtracks": 1,
        "external_nodal_work_trapezoid_j": 1.0e-3,
    }
    record.update(overrides)
    return record


def _substep_without_work(start, end, **overrides):
    record = _substep(start, end, **overrides)
    del record["external_nodal_work_trapezoid_j"]
    return record


def _continuation(substeps, **overrides):
    block = {
        "mode": "declared_physical_dyadic_substeps",
        "revision": _CONTACT_PATH_PHYSICAL_SUBSTEPS_REVISION,
        "cutback_reason": "contact_semismooth_stagnation",
        "maximum_halvings": 6,
        "substeps": substeps,
        "halvings_used": 1,
        "trial_count": 2,
        "accepted_substep_count": len(substeps),
        "stagnation_patience": 2,
        "stagnation_fraction": 0.125,
        "stagnation_merit_rate": 0.05,
    }
    block.update(overrides)
    return block


def _record(continuation, **overrides):
    program = _StubProgram()
    record = {
        "step_index": 1,
        "target": program.target_at(1).tolist(),
        "continuation": continuation,
    }
    record.update(overrides)
    return record


def _validate(record, maximum_halvings=6, program=None):
    settings = _StubSettings(maximum_halvings=maximum_halvings)
    _validate_continuation_history_record(
        record, program if program is not None else _StubProgram(), settings)


def _growth_chain_substeps(maximum_halvings):
    """Producer-reachable deepest-span growth chain back to the 0.5 cap.

    The producer halves from 0.5 down to 2^-M, accepts that deepest
    span, and can then only double on each acceptance, so the reachable
    way back to a 0.5 span is the full dyadic ladder 2^-M .. 2^-1.
    """
    substeps = []
    tau = 0.0
    for exponent in range(maximum_halvings, 0, -1):
        span = 2.0 ** -exponent
        substeps.append(_substep(tau, tau + span))
        tau += span
    substeps.append(_substep(tau, 1.0))
    return substeps


def test_reader_accepts_producer_reachable_chronology():
    # Interior spans are exact dyadic unit fractions 2^-k, k <= 6; the
    # final span reaches the macro endpoint inside the 0.5 cap. One
    # failure before the second substep: halvings_used = 1 + 4 - 3 = 2.
    _validate(_record(_continuation([
        _substep(0.0, 0.5),
        _substep(0.5, 0.75),
        _substep(0.75, 1.0),
    ], halvings_used=2, trial_count=4)))
    # Deepest producer span 2^-6 reached after five failures, then the
    # only reachable growth ladder back toward the cap (the deepest
    # span cannot repeat: a further failure at the budget limit would
    # exhaust the halvings). Five failures, seven accepted substeps,
    # halvings_used = 1 + 12 - 7 = 6.
    _validate(_record(_continuation(
        _growth_chain_substeps(6), halvings_used=6, trial_count=12)))


def test_reader_accepts_repeated_final_interval_failure():
    # The final truncated interval (0.75, 1.0) is retried unchanged
    # after a failure while the halved nominal span 0.25 still covers
    # it: a producer-permitted failure that must not be forbidden.
    _validate(_record(_continuation([
        _substep(0.0, 0.5),
        _substep(0.5, 0.75),
        _substep(0.75, 1.0),
    ], halvings_used=3, trial_count=5)))


def test_reader_accepts_non_power_of_two_final_span():
    # The producer's final span is 1 - tau for the reaching trial: any
    # positive multiple of 2^-M up to the 0.5 cap, not a power of two.
    # Reachable shape: 0.25 accepted after one failure, 0.125 after two
    # more, then growth 0.25 and the truncated final 0.375.
    _validate(_record(_continuation([
        _substep(0.0, 0.25),
        _substep(0.25, 0.375),
        _substep(0.375, 0.625),
        _substep(0.625, 1.0),
    ], halvings_used=4, trial_count=7)))


def test_reader_rejects_span_below_producer_minimum():
    # 2^-7 interior span: admissible on the old 2^(M+1) grid, below the
    # producer's 2^-6 cumulative halving minimum.
    with pytest.raises(ValueError, match="chronology is invalid"):
        _validate(_record(_continuation([
            _substep(0.0, 0.0078125),
            _substep(0.0078125, 0.5),
            _substep(0.5, 1.0),
        ], halvings_used=2, trial_count=4)))


def test_reader_rejects_non_dyadic_unit_interior_span():
    # 3*2^-6 interior span: an integer multiple on the M grid (and on
    # the old grid) but the producer only emits powers of two.
    with pytest.raises(ValueError, match="chronology is invalid"):
        _validate(_record(_continuation([
            _substep(0.0, 0.046875),
            _substep(0.046875, 0.5),
            _substep(0.5, 1.0),
        ], halvings_used=2, trial_count=4)))


def test_reader_rejects_oversized_final_span():
    # A single full-span substep: the producer caps every span at 0.5.
    with pytest.raises(ValueError, match="chronology is invalid"):
        _validate(_record(_continuation(
            [_substep(0.0, 1.0)], halvings_used=1, trial_count=1)))


def test_reader_rejects_interior_span_growth_beyond_producer_doubling():
    # Reviewer counterexample: a 2^-6 accepted span followed by a 0.5
    # accepted span. Reaching 0.5 again needs the intermediate accepted
    # substeps 2^-5 .. 2^-1; the producer doubles at most once per
    # acceptance, so this chronology is unreachable even though every
    # span is individually valid and the accounting identity holds
    # (halvings_used = 1 + 8 - 3 = 6).
    with pytest.raises(ValueError, match="chronology is invalid"):
        _validate(_record(_continuation([
            _substep(0.0, 0.015625),
            _substep(0.015625, 0.515625),
            _substep(0.515625, 1.0),
        ], halvings_used=6, trial_count=8)))


def test_reader_rejects_halvings_identity_mismatch():
    # The producer never resets its halvings counter:
    # halvings_used = 1 + trial_count - accepted_substep_count exactly.
    substeps = [_substep(0.0, 0.5), _substep(0.5, 1.0)]
    with pytest.raises(ValueError, match="accounting is invalid"):
        _validate(_record(_continuation(
            substeps, halvings_used=1, trial_count=3)))
    with pytest.raises(ValueError, match="accounting is invalid"):
        _validate(_record(_continuation(
            substeps, halvings_used=2, trial_count=4)))


def test_reader_rejects_halvings_inconsistent_with_spans():
    substeps = [_substep(0.0, 0.5), _substep(0.5, 1.0)]
    # Identity-consistent (1 + 3 - 2 = 2) but unproducible: the extra
    # failure would halve the nominal span below the recorded 0.5 final
    # span, so the final interval could not have been accepted.
    with pytest.raises(ValueError, match="chronology is invalid"):
        _validate(_record(_continuation(
            substeps, halvings_used=2, trial_count=3)))
    # A 2^-6 span requires at least five failures before it; the growth
    # chain implies exactly five, so halvings_used = 3 (identity-consistent
    # 1 + 9 - 7) is unproducible.
    with pytest.raises(ValueError, match="chronology is invalid"):
        _validate(_record(_continuation(
            _growth_chain_substeps(6), halvings_used=3, trial_count=9)))


def test_reader_rejects_work_key_program_mismatch():
    # The producer emits external_nodal_work_trapezoid_j on a substep
    # exactly when the program declares nodal forces.
    with pytest.raises(ValueError, match="substep is invalid"):
        _validate(_record(_continuation([
            _substep(0.0, 0.5),
            _substep(0.5, 1.0),
        ])), program=_StubProgram(nodal_forces=()))
    with pytest.raises(ValueError, match="substep is invalid"):
        _validate(_record(_continuation([
            _substep(0.0, 0.5, external_nodal_work_trapezoid_j=None),
            _substep(0.5, 1.0),
        ])))
    with pytest.raises(ValueError, match="substep is invalid"):
        _validate(_record(_continuation([
            _substep_without_work(0.0, 0.5),
            _substep(0.5, 1.0),
        ])))
    # Signed finite work is admissible in both directions, and a
    # force-free program must carry no work key at all.
    _validate(_record(_continuation([
        _substep(0.0, 0.5, external_nodal_work_trapezoid_j=-0.5),
        _substep(0.5, 1.0, external_nodal_work_trapezoid_j=0.0),
    ])))
    _validate(_record(_continuation([
        _substep_without_work(0.0, 0.5),
        _substep_without_work(0.5, 1.0),
    ])), program=_StubProgram(nodal_forces=()))


def test_reader_rejects_stagnation_settings_mismatch():
    # Recorded stagnation controls must equal the live settings of the
    # validating run, not merely sit inside valid ranges.
    substeps = [_substep(0.0, 0.5), _substep(0.5, 1.0)]
    with pytest.raises(ValueError, match="accounting is invalid"):
        _validate(_record(_continuation(substeps, stagnation_patience=3)))
    with pytest.raises(ValueError, match="accounting is invalid"):
        _validate(_record(_continuation(substeps, stagnation_fraction=0.25)))
    with pytest.raises(ValueError, match="accounting is invalid"):
        _validate(_record(_continuation(substeps, stagnation_merit_rate=0.06)))
    # Exact live settings pass through.
    _validate(_record(_continuation(substeps)), maximum_halvings=6)


def test_reader_rejects_negative_and_nonfinite_diagnostics():
    with pytest.raises(ValueError, match="substep is invalid"):
        _validate(_record(_continuation([
            _substep(0.0, 0.5, iterations=-1),
            _substep(0.5, 1.0),
        ])))
    with pytest.raises(ValueError, match="substep is invalid"):
        _validate(_record(_continuation([
            _substep(0.0, 0.5, residual_norm=float("nan")),
            _substep(0.5, 1.0),
        ])))
    with pytest.raises(ValueError, match="substep is invalid"):
        _validate(_record(_continuation([
            _substep(0.0, 0.5, candidate_evaluations=-2),
            _substep(0.5, 1.0),
        ])))
    with pytest.raises(ValueError, match="substep is invalid"):
        _validate(_record(_continuation([
            _substep(0.0, 0.5, complementarity_relative_force=-1.0e-7),
            _substep(0.5, 1.0),
        ])))
    with pytest.raises(ValueError, match="substep is invalid"):
        _validate(_record(_continuation([
            _substep(0.0, 0.5, external_nodal_work_trapezoid_j=float("inf")),
            _substep(0.5, 1.0),
        ])))


def test_reader_rejects_incompatible_recorded_halvings_budget():
    # Rehashed block claiming a different halvings budget than the
    # enabled settings: must fail closed even with a valid chronology.
    with pytest.raises(ValueError, match="halvings budget is incompatible"):
        _validate(_record(_continuation([
            _substep(0.0, 0.5),
            _substep(0.5, 1.0),
        ], maximum_halvings=12)))
    with pytest.raises(ValueError, match="halvings budget is incompatible"):
        _validate(_record(_continuation([
            _substep(0.0, 0.5),
            _substep(0.5, 1.0),
        ], maximum_halvings=0)))


def test_reader_rejects_out_of_budget_accounting():
    substeps = [_substep(0.0, 0.5), _substep(0.5, 1.0)]
    # Trial count beyond the producer's recorded budget 2^(M+1)+2M+2.
    with pytest.raises(ValueError, match="accounting is invalid"):
        _validate(_record(_continuation(substeps, trial_count=1000)))
    # halvings_used beyond the recorded budget.
    with pytest.raises(ValueError, match="accounting is invalid"):
        _validate(_record(_continuation(substeps, halvings_used=7)))
    # Accepted count disagreeing with the declared substep list.
    with pytest.raises(ValueError, match="accounting is invalid"):
        _validate(_record(_continuation(substeps, accepted_substep_count=3)))
    # Missing accounting fields after a rehash.
    block = _continuation(substeps)
    del block["halvings_used"]
    with pytest.raises(ValueError, match="accounting is invalid"):
        _validate(_record(block))
    # Invalid recorded stagnation settings.
    with pytest.raises(ValueError, match="accounting is invalid"):
        _validate(_record(_continuation(substeps, stagnation_fraction=0.0)))
    with pytest.raises(ValueError, match="accounting is invalid"):
        _validate(_record(_continuation(substeps, stagnation_patience=0)))
    with pytest.raises(ValueError, match="accounting is invalid"):
        _validate(_record(_continuation(substeps, stagnation_merit_rate=1.0)))


def test_reader_rejects_invalid_cutback_reason_and_endpoint():
    substeps = [_substep(0.0, 0.5), _substep(0.5, 1.0)]
    with pytest.raises(ValueError, match="cutback reason is invalid"):
        _validate(_record(_continuation(substeps, cutback_reason="")))
    with pytest.raises(ValueError, match="cutback reason is invalid"):
        block = _continuation(substeps)
        del block["cutback_reason"]
        _validate(_record(block))
    # Target no longer matching the program's own waypoint.
    with pytest.raises(ValueError, match="endpoint is inconsistent"):
        _validate(_record(_continuation(substeps), target=[0.0, 0.0]))


def test_reader_accepts_minimum_budget_chronology():
    # M=1: the only producer-reachable spans are 0.5.
    _validate(_record(_continuation([
        _substep(0.0, 0.5),
        _substep(0.5, 1.0),
    ], maximum_halvings=1, halvings_used=1, trial_count=2)),
        maximum_halvings=1)
    # M=12: the deepest producer span 2^-12 is reachable, and the only
    # reachable way back to the cap is the full dyadic growth ladder
    # (eleven failures, thirteen accepted substeps, halvings_used = 12).
    _validate(_record(_continuation(
        _growth_chain_substeps(12),
        maximum_halvings=12, halvings_used=12, trial_count=24)),
        maximum_halvings=12)


def test_reader_ignores_absent_continuation_block():
    # Disabled-run records carry no continuation block and must pass
    # through untouched (macro-only restart compatibility).
    program = _StubProgram()
    _validate_continuation_history_record(
        {"step_index": 1, "target": program.target_at(1).tolist()},
        program, _StubSettings())
