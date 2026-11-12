"""Allocation/fixture admission only; no nonlinear assembly or solve."""
from types import SimpleNamespace
import numpy as np
import pytest

from anyfesolver import _ge_beam3_retained_fibre_state as state
from anyfesolver._ge_beam3_seeded_load_program import ForceProgram
from docs.reference_cases import ge_beam3_fibre_arch_refinement as family


def test_twelve_macro_layout_is_bounded_without_mechanical_assembly():
    model = family.model(12)
    program = ForceProgram((0.,), ((13, 0., -1., 0.),))
    with pytest.raises(ValueError, match='bounded standalone'): state.Layout(model, program)
    layout = state.Layout(model, program, max_coordinates=512)
    assert state.MAX_RETAINED_COORDINATES == 256
    assert state.ADMITTED_DENSE_COORDINATE_BUDGETS == (256, 512)
    assert max(state.ADMITTED_DENSE_COORDINATE_BUDGETS)**2*np.dtype(np.float64).itemsize == 2*(1 << 20)
    assert layout.nodal_count == 150 and layout.count == 438
    assert len(layout.free) == 426 and len(layout.slots) == 12
    assert max(max(row) for row in layout.slots) == 437
    assert all(len(row) == 42 for row in layout.slots)
    assert len(layout.initial.positions) == 25
    layout.guard()


@pytest.mark.parametrize('count', [513, 1024, 10**9])
def test_oversized_layout_refuses_before_constraint_or_dense_allocation(monkeypatch, count):
    program = ForceProgram((0.,))
    model = SimpleNamespace(mesh=SimpleNamespace(elements={1: object()},
        dof_manager=SimpleNamespace(total_dofs=count-24)))
    def forbidden(*args, **kwargs): pytest.fail('allocation reached before size rejection')
    monkeypatch.setattr(state, 'build_constraint_transformation', forbidden)
    monkeypatch.setattr(state.np, 'zeros', forbidden)
    monkeypatch.setattr(state.sparse, 'eye', forbidden)
    with pytest.raises(ValueError, match='bounded standalone'): state.Layout(model, program, max_coordinates=512)


@pytest.mark.parametrize('budget', [True, 256., 0, 257, 1024, None])
def test_unregistered_allocation_budget_rejected_before_model_access(budget):
    with pytest.raises(ValueError, match='admitted dense coordinate budget'):
        state.Layout(object(), ForceProgram((0.,)), max_coordinates=budget)


def test_default_capture_is_unchanged_and_explicit_budget_is_bound():
    model = family.model(6); program = ForceProgram((0.,), ((7, 0., -1., 0.),))
    default = state.Layout(model, program)
    explicit = state.Layout(model, program, max_coordinates=256)
    expanded = state.Layout(model, program, max_coordinates=512)
    assert default.identity == explicit.identity
    assert expanded.identity != default.identity
    np.testing.assert_array_equal(expanded.reference_positions, default.reference_positions)
    np.testing.assert_array_equal(expanded.reference_frames, default.reference_frames)
    assert expanded.slots == default.slots and expanded.free == default.free
    expanded.max_coordinates = 256
    with pytest.raises(ValueError, match='frozen model/program changed'): expanded.guard()


@pytest.mark.parametrize('entry', ['force', 'translation', 'seeded', 'solve'])
def test_explicit_budget_reaches_layout_before_any_mechanics(monkeypatch, entry):
    from anyfesolver import _ge_beam3_retained_fibre_control as translation
    from anyfesolver import _ge_beam3_seeded_fibre_control as seeded
    class CapturedBeforeAssembly(Exception): pass
    def capture(model, program, *, max_coordinates):
        assert max_coordinates == 512
        raise CapturedBeforeAssembly
    monkeypatch.setattr(state, 'Layout', capture)
    calls = {
        'force': lambda: state.Context(object(), ForceProgram((0.,)), max_coordinates=512),
        'translation': lambda: translation.Context(object(), family.program(12), max_coordinates=512),
        'seeded': lambda: seeded.Context(object(), family.program(12), max_coordinates=512),
        'solve': lambda: seeded.solve_translation_program(object(), family.program(12), max_coordinates=512),
    }
    with pytest.raises(CapturedBeforeAssembly): calls[entry]()
