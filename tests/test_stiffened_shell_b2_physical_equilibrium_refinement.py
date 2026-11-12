"""Bounded correction of an inaccurate physical-footprint KKT solve."""

import numpy as np
import pytest

from anyfesolver import stiffened_shell as api
from examples.stiffened_shell_b2_physical_rigid_footprint import definition as physical_definition


class _InaccurateFirstSolve:
    def __init__(self, original, physical_scale, *, refuse_correction=False):
        self.original = original
        self.physical_scale = physical_scale
        self.refuse_correction = refuse_correction
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.original, name)

    def solve(self, rhs):
        result = self.original.solve(rhs)
        self.calls += 1
        if self.calls == 1:
            result = result.copy()
            result[0] += 1e-6/self.physical_scale[0]
        elif self.refuse_correction:
            return np.zeros_like(result)
        return result


def _load(owner):
    applied = np.zeros(owner._u0.shape)
    applied[13] = 1.
    return applied


def test_accepted_physical_solve_keeps_one_solve():
    owner = api.create_analysis(definition=physical_definition())
    try:
        result = owner.solve_linear(_load(owner))
        assert result.status == "completed", result.failure_reason
        assert result.factorization_count == result.solve_count == 1
    finally:
        owner.close()


def test_inaccurate_physical_solve_is_corrected_with_same_factorization():
    owner = api.create_analysis(definition=physical_definition())
    try:
        original = owner._handle
        owner._handle = _InaccurateFirstSolve(original, owner._factor_scale)
        result = owner.solve_linear(_load(owner))
        assert result.status == "completed", result.failure_reason
        assert result.factorization_count == 1
        assert result.solve_count == 2
        assert owner._handle.calls == 2
        assert original.solve_count == 2
    finally:
        owner.close()


def test_nonimproving_physical_correction_is_typed_failure():
    owner = api.create_analysis(definition=physical_definition())
    try:
        original = owner._handle
        owner._handle = _InaccurateFirstSolve(
            original, owner._factor_scale, refuse_correction=True)
        result = owner.solve_linear(_load(owner))
        assert result.status == "failed"
        assert "correction did not improve" in result.failure_reason
        assert result.factorization_count == 1
        assert result.solve_count == 2
    finally:
        owner.close()


def test_nonfinite_physical_correction_is_typed_failure():
    owner = api.create_analysis(definition=physical_definition())
    try:
        original = owner._handle
        corrupt = _InaccurateFirstSolve(original, owner._factor_scale)
        solve = corrupt.solve
        def nonfinite(rhs):
            value = solve(rhs)
            if corrupt.calls == 2:
                value[:] = np.nan
            return value
        corrupt.solve = nonfinite
        owner._handle = corrupt
        result = owner.solve_linear(_load(owner))
        assert result.status == "failed"
        assert "finite panel equilibrium correction" in result.failure_reason
        assert result.factorization_count == 1 and result.solve_count == 2
    finally:
        owner.close()


def test_legacy_nodal_footprint_does_not_enter_refinement(monkeypatch):
    source = physical_definition()
    nodal = api.B2NodalRigidFootprint(
        "nodal-left", (1, 2, 5, 4), 10, ((1, 2, 5, 4),))
    legacy = api.B2Q4PanelDefinition(
        source.profile, source.shell_nodes_m, source.shells, source.members,
        (), source.transformation, source.prescribed_offset, (nodal,))
    owner = api.create_analysis(definition=legacy)
    try:
        def forbidden(*_args):
            pytest.fail("legacy route entered physical equilibrium refinement")
        monkeypatch.setattr(owner, "_correct_physical_equilibrium", forbidden)
        result = owner.solve_linear(_load(owner))
        assert result.status == "completed", result.failure_reason
        assert result.factorization_count == result.solve_count == 1
    finally:
        owner.close()
