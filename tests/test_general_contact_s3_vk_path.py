"""S3 lifted path under the von Karman law (fails on the reference-linear adapter)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import test_general_contact_event_subdivision as events  # noqa: E402
from anyfesolver.dynamics import TransientConfig  # noqa: E402
from anyfesolver.general_contact import transient as driver  # noqa: E402


def _tilted(stop=.012):
    model, definition, initial = events._guided_s3_descriptor_impact(tilt=.002, span=.1)
    return driver.solve_transient_contact(
        model, definition, TransientConfig(.0002, stop, initial_velocity=initial),
        solver_controls=events.CONTROLS)


def test_tilted_s3_lifted_path_is_certified_against_the_von_karman_force():
    # The reference-linear adapter refused this step ('endpoint mismatch exceeds its
    # numerical budget'): the von Karman excess force is ~1e-7 N, the budget 1e-8.
    result = _tilted()
    assert len(result.localized_contact_event_times) >= 2
    assert all(g >= -.001 for g in result.swept_minimum_gaps)


def test_unconverged_interior_force_model_is_refused(monkeypatch):
    # Dropping the curvature leaves the chord model, whose measured interior position
    # error (1.2e-10 m) exceeds the 1e-10 m path budget: the guard must stay closed.
    monkeypatch.setattr(driver._S3AffinePathAdapter, "force_model",
                        staticmethod(lambda ends, middle: (ends[0], ends[1]-ends[0], np.zeros_like(middle))))
    with pytest.raises(ValueError, match="interior disagrees with von Karman"):
        _tilted()
