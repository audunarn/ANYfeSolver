"""Pair-only compatibility regression for the public solver-owned operation."""
import numpy as np
import pytest

from anyfesolver_results import shell_surface_component


@pytest.mark.parametrize('surface', ['top', 'bottom'])
@pytest.mark.parametrize('component', ['xx', 'yy', 'xy'])
def test_pair_only_surface_keeps_original_words_and_inputs(surface, component):
    membrane = np.array([0., -0., 1e16, 1e-300, -4.], dtype=np.float64)
    bending = np.array([-0., 0., -1e16, -1e-300, 2.], dtype=np.float64)
    before = membrane.copy(), bending.copy()
    # Independent original application expression, without unrelated components.
    expected = membrane + bending if surface == 'top' else membrane - bending
    actual = shell_surface_component({f'membrane_{component}': membrane,
                                     f'bending_{component}': bending}, surface, component)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    np.testing.assert_array_equal(membrane.view(np.uint64), before[0].view(np.uint64))
    np.testing.assert_array_equal(bending.view(np.uint64), before[1].view(np.uint64))
    assert not np.shares_memory(actual, membrane)


def test_scalar_broadcast_empty_and_missing_samples_are_preserved():
    assert shell_surface_component({'membrane_xx': 4, 'bending_xx': 2}, 'bottom', 'xx') == 2
    values = shell_surface_component({'membrane_yy': [1., np.nan], 'bending_yy': 2.}, 'top', 'yy')
    assert values.dtype == np.float64 and values[0] == 3 and np.isnan(values[1])
    assert shell_surface_component({'membrane_xy': [], 'bending_xy': []}, 'top', 'xy').size == 0


@pytest.mark.parametrize('record,surface,component', [
    ({}, 'top', 'xx'), ({'membrane_xx': 1}, 'top', 'xx'),
    ({'membrane_xx': 1, 'bending_xx': None}, 'top', 'xx'),
    ({}, 'middle', 'xx'), ({}, 'top', 'xz'),
])
def test_missing_or_invalid_pair_refuses_without_fabrication(record, surface, component):
    with pytest.raises(ValueError):
        shell_surface_component(record, surface, component)
