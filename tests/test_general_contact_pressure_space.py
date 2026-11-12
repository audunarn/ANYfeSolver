"""Independent matrix controls for pressure-space rank reporting."""

import numpy as np
import pytest

from anyfesolver.general_contact.pressure_space import supported_pressure_gram


def test_area_scaling_does_not_destroy_an_independent_pressure_mode():
    gram = np.diag([1.0e-24, 1.0])
    assert np.linalg.matrix_rank(gram) == 1
    result = supported_pressure_gram(gram, np.array([1.0e-21, 1.0]))
    assert result.supported_modes == 2
    assert result.scaled_rank == 2
    assert result.scaled_condition == pytest.approx(1.0)


def test_true_dependence_and_zero_support_remain_visible():
    gram = np.array([[1.0, 1.0, 0.0],
                     [1.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0]])
    result = supported_pressure_gram(gram, np.array([1.0, 1.0, 0.0]))
    assert result.supported_modes == 2
    assert result.scaled_rank == 1
    assert result.scaled_condition is None


def test_materially_indefinite_supported_gram_is_rejected():
    with pytest.raises(ValueError, match="negative scaled eigenvalue"):
        supported_pressure_gram(np.array([[1.0, 2.0], [2.0, 1.0]]),
                                np.ones(2))
