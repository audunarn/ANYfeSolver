"""Consequential S3 admission and actual generalized recovery checks."""
from dataclasses import fields, replace
import numpy as np
import pytest
from anyfesolver import stiffened_shell as ss
from test_stiffened_shell_native_point_displacement import (
    model, owner, ACTIONS, DIRECTION, CONTROL, OPTIONS, HISTORICAL_POLICY)
from test_stiffened_shell_ge_finite_experimental import _q4_shell_j2_panel


def test_native_s3_shell_j2_displacement_is_atomic(request):
    base = _q4_shell_j2_panel()
    definition = replace(base, shell=replace(base.shell,
        q4_elements=((1, 2, 5, 4),), s3_elements=((2, 3, 6), (2, 6, 5))),
        q4_pullback_policy=ss.Q4_ALGORITHMIC_PULLBACK_V1)
    analysis = owner(request, definition)
    assert analysis.capabilities.get('displacement_control') != 'experimental_native_point_elastic_s3_fixed_translation_v1'
    with pytest.raises(ValueError):
        analysis.solve(DIRECTION, (.001,), displacement_control=CONTROL, **OPTIONS)
    assert analysis._generation == 0 and analysis._finite_driver is None
    assert analysis._accepted is None


def compare_generalized(left, right):
    assert left and len(left) == len(right)
    for a, b in zip(left, right, strict=True):
        assert (a.element_id, a.recovery_policy, a.production_qualified) == (b.element_id, b.recovery_policy, b.production_qualified)
        for s, t in zip(a.stations, b.stations, strict=True):
            for field in fields(s):
                x, y = getattr(s, field.name), getattr(t, field.name)
                if isinstance(x, np.ndarray):
                    np.testing.assert_array_equal(x, y)
                else:
                    assert x == y


def test_native_s3_generalized_recovery_restore_continue(request):
    base = model('generalized')
    definition = replace(base, shell=replace(base.shell, q4_elements=(),
        s3_elements=((1, 2, 5), (1, 5, 4), (2, 3, 6), (2, 6, 5))),
        q4_pullback_policy=HISTORICAL_POLICY)
    analysis = owner(request, definition)
    resumed = owner(request, definition)
    analysis.solve(ACTIONS, (.5, 1.), **OPTIONS)
    dof = analysis._shell_count + 6*analysis._beam_index[105] + 2
    initial = analysis._finite_driver._state.q[dof]
    kwargs = dict(OPTIONS, displacement_control=CONTROL)
    prefix = analysis.solve(DIRECTION, (initial*.99,), **kwargs)
    restored = resumed.restore(analysis.checkpoint())
    compare_generalized(prefix.beam_generalized_material, restored.beam_generalized_material)
    final = analysis.solve(DIRECTION, (initial*1.01,), **kwargs)
    continued = resumed.solve(DIRECTION, (initial*1.01,), **kwargs)
    compare_generalized(final.beam_generalized_material, continued.beam_generalized_material)
    assert analysis.checkpoint() == resumed.checkpoint()
