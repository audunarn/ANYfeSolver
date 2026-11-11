"""Exercise construction work reuse through real owner candidate transactions."""
import numpy as np
import pytest

from anyfesolver.control import CancellationToken
from test_general_contact_native_curved_s3_static import _analysis


def owner():
    return _analysis(targets=(1.,), physical_fibres=True, plastic_fibres=True)


def test_initial_candidate_reuses_preflight_and_changed_pose_recomputes():
    analysis = owner()
    initial = analysis.initial
    checkpoint = analysis._checkpoint(())
    count = analysis.pair_evaluations
    with analysis.context.operation(None):
        analysis._begin_candidate_context(None)
        try:
            same = analysis._contact(initial.mechanical, initial.shell_displacement)
            assert same is initial.contact
            assert analysis.pair_evaluations == count
            changed = initial.shell_displacement.copy()
            changed[analysis.shell_free[0]] += 1e-8
            result = analysis._contact(initial.mechanical, changed)
            assert result is not same
            assert analysis.pair_evaluations == count + len(analysis.pairs)
        finally:
            analysis._end_candidate_context()
    assert analysis._contact_reuse_entry is None
    assert not analysis._contact_reuse_enabled
    assert analysis._checkpoint(()) == checkpoint


def test_initial_cached_work_mutation_fails_before_reuse():
    analysis = owner()
    original = analysis.initial.contact.residual.copy()
    count = analysis.pair_evaluations
    # Deliberate corruption of a caller-owned runtime buffer, not a fixture.
    analysis.initial.contact.residual.flags.writeable = True
    analysis.initial.contact.residual[0] += 1.
    with analysis.context.operation(None):
        analysis._begin_candidate_context(None)
        try:
            with pytest.raises(ValueError, match='cached native/S3 contact work mutated'):
                analysis._contact(analysis.initial.mechanical,
                                  analysis.initial.shell_displacement)
            assert analysis.pair_evaluations == count
            assert analysis._contact_reuse_entry is None
        finally:
            analysis._end_candidate_context()
    np.testing.assert_array_equal(analysis.initial.contact.residual[1:], original[1:])


def test_precancelled_driver_preserves_initial_checkpoint_and_cleans_reuse():
    analysis = owner()
    checkpoint = analysis._checkpoint(())
    count = analysis.pair_evaluations
    token = CancellationToken()
    token.cancel('before candidate')
    result = analysis.solve(cancellation_token=token)
    assert result.status == 'cancelled'
    assert result.checkpoint == checkpoint
    assert analysis.pair_evaluations == count
    assert analysis._contact_reuse_entry is None
    assert not analysis._contact_reuse_enabled
