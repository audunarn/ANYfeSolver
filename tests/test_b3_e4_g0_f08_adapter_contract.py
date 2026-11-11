"""F08 typed native local adapter parity and accepted-predecessor ownership."""

from hashlib import sha256
from dataclasses import replace
import json

import numpy as np
import pytest

from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver._ge_beam3_retained_fibre_program import solve_force_program
from anyfesolver._ge_beam3_retained_generalized_program import solve as solve_generalized
from anyfesolver._ge_beam3_retained_fibre_state import Context as FibreContext
from anyfesolver._ge_beam3_retained_generalized_state import (
    Context as GeneralizedContext, Program as GeneralizedProgram,
)
from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_seeded_load_program import ForceProgram
from anyfesolver._stiffened_shell_b3_local_adapter import (
    evaluate_local_trial, evaluate_sparse_trial,
)
from scripts._b3_e4_f03_native_fixture import build
from scripts.b3_e4_f03_square_section_control import square_torsion_constant
from test_b3_e4_g0_f08_local_operator_seam import _sparse_local_gather
from test_ge_beam3_retained_fibre import make_model


def _relative(a, b):
    return np.linalg.norm(a-b)/max(1., np.linalg.norm(b))


def _spatial_fibre_dense(layout, residual, chart_hessian, parameter):
    """Independent global conversion of the dense owner's Exp Hessian."""
    result = np.array(chart_hessian, copy=True)
    starts = [6*i+3 for i in range(len(layout.node_ids))]
    for i in range(len(layout.elements)):
        starts.extend((layout.nodal_count+24*i,
                       layout.nodal_count+24*i+3))
    for start in starts:
        moment = np.array(residual[start:start+3], copy=True)
        if start < layout.nodal_count:
            # The fibre owner's dense residual has the applied nodal load
            # subtracted after element evaluation.  Its Exp Hessian is
            # internal, so the moment correction must use the internal row.
            moment += parameter*layout.force[start:start+3]
        skew = np.array(((0., -moment[2], moment[1]),
                         (moment[2], 0., -moment[0]),
                         (-moment[1], moment[0], 0.)))
        result[start:start+3, start:start+3] -= .5*skew
    return result


def _block_inventory(context, state, origin, parameter, family, assembled):
    """Direct current-adapter 18 physical + 24 retained incidence checks."""
    layout = context.layout if family == "physical_fibre" else context
    assert len(layout.elements) == 2
    shared = set(layout.slots[0]) & set(layout.slots[1])
    retained = [set(slots[18:]) for slots in layout.slots]
    assert len(shared) == 6 and all(len(slots) == 24 for slots in retained)
    assert not (retained[0] & retained[1])
    assert not (shared & (retained[0] | retained[1]))
    norms = []
    for index in range(2):
        local = evaluate_local_trial(context, state, origin, parameter,
                                     family=family, element_index=index)
        positions = [row for row, slot in enumerate(local.slots[:18]) if slot in shared]
        assert len(positions) == 6
        measured = [float(np.linalg.norm(block)) for block in (
            local.tangent[:18, 18:], local.tangent[18:, :18],
            local.tangent[18:, 18:], local.residual[positions],
            local.tangent[np.ix_(positions, positions)])]
        assert min(measured) > 1e-12, measured
        norms.append(measured)
    a, b = (np.array(sorted(slots), dtype=int) for slots in retained)
    assert assembled.tangent[a, :][:, b].nnz == 0
    assert assembled.tangent[b, :][:, a].nnz == 0
    return norms


def test_current_adapter_uses_accepted_generalized_predecessor_and_retained_blocks(tmp_path):
    fixture = build(4, z_offset=.0675, torsion_constant=square_torsion_constant())
    pattern = DistributedPattern(
        LinePattern(tuple((eid, .09, -.03, .02) for eid in fixture.model.mesh.elements)),
        tuple((eid, .07, -.04, .05) for eid in fixture.model.mesh.elements))
    program = GeneralizedProgram((.5, 1.), pattern)
    accepted = solve_generalized(fixture.model, program, stop_after=1)
    assert accepted.status == "paused", accepted.failure
    assert accepted.completed_targets == 1
    context = GeneralizedContext(fixture.model, program)
    checkpoint_hash = sha256(accepted.checkpoint).hexdigest()
    restored, _ = context.restore(accepted.checkpoint, expected_sha256=checkpoint_hash)
    before = canonical((restored.mechanical.descriptor(), restored.histories))
    direction = np.zeros(context.count)
    direction[context.free] = 2e-4*np.sin(np.asarray(context.free)+.3)
    trial = context.advance(restored.mechanical, direction)
    assert np.linalg.norm(trial.nodal_frames-context.reference_frames) > 1e-7
    assert np.linalg.norm(trial.cell_rotations-np.eye(3)) > 1e-7
    assert np.linalg.norm(trial.resultants) > 1e-7
    dense = context.assemble(trial, .5, restored.histories)
    assert dense[2][1] > 1e-7
    result = evaluate_sparse_trial(context, trial, restored.histories, .5,
                                  family="generalized")
    residual_error = float(np.max(np.abs(result.residual-dense[0])))
    tangent_error = float(np.max(np.abs(result.tangent.toarray()-dense[1])))
    normalized_residual = float(_relative(result.residual, dense[0]))
    normalized_tangent = float(_relative(result.tangent.toarray(), dense[1]))
    assert max(residual_error, tangent_error) <= 1e-9
    assert max(normalized_residual, normalized_tangent) <= 1e-11
    assert tuple(canonical(h) for h in result.candidate_histories) == tuple(
        canonical(response.history) for response in dense[3])
    norms = _block_inventory(context, trial, restored.histories, .5,
                             "generalized", result)
    assert canonical((restored.mechanical.descriptor(), restored.histories)) == before
    assert sha256(accepted.checkpoint).hexdigest() == checkpoint_hash
    (tmp_path/"accepted-generalized-blocks.json").write_text(json.dumps({
        "residual_absolute_error": residual_error, "tangent_absolute_error": tangent_error,
        "residual_normalized_error": normalized_residual,
        "tangent_normalized_error": normalized_tangent,
        "compatibility": float(dense[2][1]), "local_block_norms": norms,
        "count": context.count, "nnz": result.tangent.nnz,
        "accepted_checkpoint_sha256": checkpoint_hash,
    }, indent=2))


@pytest.mark.parametrize("family", ("generalized", "physical_fibre"))
@pytest.mark.parametrize("trial", (False, True))
def test_typed_adapter_matches_old_local_seam_and_dense_owner(
        family, trial, monkeypatch):
    if family == "generalized":
        context = build(4, z_offset=.0675,
                        torsion_constant=square_torsion_constant()).context
        origin = context.initial.origins
        state = context.initial.mechanical
        layout = context
    else:
        context = FibreContext(
            make_model(contrast=1.),
                ForceProgram((.25,), ((5, .4, -.005, 0.),),
                         max_iterations=24))
        origin = context.initial.origins
        state = context.initial.mechanical
        layout = context.layout
    if trial:
        delta = np.zeros(layout.count)
        delta[layout.nodal_count-6+1] = 1e-6
        delta[layout.nodal_count-6+3] = 2e-6
        delta[layout.nodal_count] = 1e-6
        state = layout.advance(state, delta)
    before = canonical(origin)
    old_r, old_k, old_histories = _sparse_local_gather(
        context, state, origin, .25, fibre=family == "physical_fibre")
    dense = context.assemble(state, .25, origin)
    local = evaluate_local_trial(context, state, origin, .25,
                                 family=family, element_index=0)
    local_again = evaluate_local_trial(context, state, origin, .25,
                                       family=family, element_index=0)
    assert local.residual.shape == (42,) and local.tangent.shape == (42, 42)
    assert local.slots.shape == (42,) and not local.production_qualified
    assert local.candidate_history is not origin[0]
    np.testing.assert_array_equal(local.residual, local_again.residual)
    local.residual[0] = np.nan
    assert np.isfinite(local_again.residual).all()
    with monkeypatch.context() as patch:
        patch.setattr(context, "assemble", lambda *_: (_ for _ in ()).throw(
            AssertionError("dense assembler called by local adapter")))
        result = evaluate_sparse_trial(context, state, origin, .25,
                                       family=family)
    assert result.family == family and not result.production_qualified
    assert len(result.candidate_histories) == 2
    assert all(candidate is not previous for candidate, previous
               in zip(result.candidate_histories, origin))
    assert _relative(result.residual, old_r) == 0.
    expected_old = (old_k.toarray() if family == "generalized" else
                    _spatial_fibre_dense(layout, old_r, old_k.toarray(), .25))
    assert _relative(result.tangent.toarray(), expected_old) <= 1e-12
    assert _relative(result.residual, dense[0]) <= 1e-12
    expected_dense = (dense[1] if family == "generalized" else
                      _spatial_fibre_dense(layout, dense[0], dense[1], .25))
    assert _relative(result.tangent.toarray(), expected_dense) <= 1e-12
    assert tuple(canonical(history) for history in result.candidate_histories) == old_histories
    assert tuple(canonical(history) for history in result.candidate_histories) == tuple(
        canonical(response.history) for response in dense[3])
    assert canonical(origin) == before


def test_typed_adapter_retains_yielded_accepted_predecessor(tmp_path):
    program = ForceProgram((.25, .5, 1., .5, 0., -.5, 0.),
                           ((5, .4, -.005, 0.),), max_iterations=24)
    accepted = solve_force_program(make_model(contrast=1.), program,
                                   stop_after=3)
    assert accepted.status == "paused" and accepted.completed_targets == 3
    checkpoint_hash = sha256(accepted.checkpoint).hexdigest()
    context = FibreContext(make_model(contrast=1.), program)
    restored, _ = context.restore(
        accepted.checkpoint, expected_sha256=checkpoint_hash)
    origin = restored.histories
    before = canonical(origin)
    state_before = canonical(restored.mechanical.descriptor())
    assert any(row[2] > 0 for cell in origin for station in cell.stations
               for row in station.rows)
    for trial in (False, True):
        state = restored.mechanical
        if trial:
            delta = np.zeros(context.layout.count)
            delta[context.layout.nodal_count-6+1] = 1e-6
            delta[context.layout.nodal_count-6+3] = 2e-6
            delta[context.layout.nodal_count] = 1e-6
            state = context.layout.advance(state, delta)
        result = evaluate_sparse_trial(context, state, origin, 1.,
                                       family="physical_fibre")
        assert all(candidate is not previous for candidate, previous
                   in zip(result.candidate_histories, origin))
        dense = context.assemble(state, 1., origin)
        assert _relative(result.residual, dense[0]) <= 1e-12
        assert _relative(result.tangent.toarray(),
                         _spatial_fibre_dense(context.layout, dense[0],
                                              dense[1], 1.)) <= 1e-12
        assert tuple(canonical(history) for history in result.candidate_histories) == tuple(
            canonical(response.history) for response in dense[3])
        assert canonical(origin) == before
        assert canonical(restored.mechanical.descriptor()) == state_before
        assert sha256(accepted.checkpoint).hexdigest() == checkpoint_hash
        if trial:
            norms = _block_inventory(context, state, origin, 1.,
                                     "physical_fibre", result)
            (tmp_path/"accepted-fibre-blocks.json").write_text(json.dumps({
                "local_block_norms": norms, "count": context.layout.count,
                "nnz": result.tangent.nnz,
                "accepted_checkpoint_sha256": checkpoint_hash,
            }, indent=2))
    direction = np.zeros(context.layout.count)
    first = context.layout.nodal_count-6
    direction[first:first+6] = (.4, -.3, .2, .15, -.1, .12)
    direction[context.layout.nodal_count:context.layout.nodal_count+7] = (
        .11, -.09, .07, .05, -.04, .03, .015)
    direction[np.asarray(context.layout.fixed, dtype=int)] = 0.
    base = evaluate_sparse_trial(context, restored.mechanical, origin, 1.,
                                 family="physical_fibre")
    expected = base.tangent@direction
    errors = []
    for step in (1e-5, 3e-6, 1e-6):
        plus_result = evaluate_sparse_trial(
            context, context.layout.advance(restored.mechanical,
                                            step*direction), origin, 1.,
            family="physical_fibre")
        minus_result = evaluate_sparse_trial(
            context, context.layout.advance(restored.mechanical,
                                            -step*direction), origin, 1.,
            family="physical_fibre")
        plus, minus = plus_result.residual, minus_result.residual
        errors.append(_relative((plus-minus)/(2*step), expected))

    def plastic_changes(candidate):
        differences = [made_row[2]-old_row[2]
                       for made_cell, old_cell in zip(candidate, origin)
                       for made_station, old_station in zip(
                           made_cell.stations, old_cell.stations)
                       for made_row, old_row in zip(
                           made_station.rows, old_station.rows)]
        return sum(value > 1e-12 for value in differences)

    # The accepted state is within this tested yield-switch interval:
    # positive perturbations activate plastic rows, negative ones remain
    # elastic. Side-specific tangents and a nearby smooth branch are
    # checked; the accepted-state semismooth tangent remains open.
    assert min(errors) > 1e-5
    assert plastic_changes(base.candidate_histories) == 0
    assert plastic_changes(plus_result.candidate_histories) > 0
    assert plastic_changes(minus_result.candidate_histories) == 0
    base_negative_error = _relative(
        expected, (base.residual-minus)/step)
    assert np.isfinite(base_negative_error)
    assert _relative(plus_result.tangent@direction,
                     (plus-base.residual)/step) <= 1e-5
    assert _relative(minus_result.tangent@direction,
                     (base.residual-minus)/step) <= 1e-5

    active_state = context.layout.advance(restored.mechanical,
                                          1e-4*direction)
    active = evaluate_sparse_trial(context, active_state, origin, 1.,
                                   family="physical_fibre")
    active_count = plastic_changes(active.candidate_histories)
    assert active_count > 0
    active_expected = active.tangent@direction
    active_errors = []
    for step in (1e-5, 3e-6, 1e-6):
        plus_result = evaluate_sparse_trial(
            context, context.layout.advance(active_state, step*direction),
            origin, 1., family="physical_fibre")
        minus_result = evaluate_sparse_trial(
            context, context.layout.advance(active_state, -step*direction),
            origin, 1., family="physical_fibre")
        assert plastic_changes(plus_result.candidate_histories) == active_count
        assert plastic_changes(minus_result.candidate_histories) == active_count
        active_errors.append(_relative(
            (plus_result.residual-minus_result.residual)/(2*step),
            active_expected))
    assert sum(error <= 1e-5 for error in active_errors) >= 2, active_errors


def test_typed_adapter_rejects_bad_family_and_parameter():
    context = build(4, z_offset=.0675,
                    torsion_constant=square_torsion_constant()).context
    with pytest.raises(ValueError, match="family"):
        evaluate_sparse_trial(context, context.initial.mechanical,
                              context.initial.origins, .25, family="legacy")
    with pytest.raises(ValueError, match="finite"):
        evaluate_sparse_trial(context, context.initial.mechanical,
                              context.initial.origins, np.nan,
                              family="generalized")


def test_typed_adapter_rejects_mutated_fibre_model_like_owner():
    model = make_model(contrast=1.)
    context = FibreContext(model, ForceProgram(
        (.25,), ((5, .4, -.005, 0.),), max_iterations=24))
    state, origin = context.initial.mechanical, context.initial.origins
    model.mesh.nodes[1].x += .001
    with pytest.raises(ValueError, match="retained fibre node/reference mismatch"):
        context.assemble(state, .25, origin)
    with pytest.raises(ValueError, match="retained fibre node/reference mismatch"):
        evaluate_local_trial(context, state, origin, .25,
                             family="physical_fibre", element_index=0)
    with pytest.raises(ValueError, match="retained fibre node/reference mismatch"):
        evaluate_sparse_trial(context, state, origin, .25,
                              family="physical_fibre")


def test_typed_adapter_rejects_invalid_generalized_state_like_owner():
    context = build(4, z_offset=.0675,
                    torsion_constant=square_torsion_constant()).context
    state, origin = context.initial.mechanical, context.initial.origins
    dof = next(dof for dof in context.fixed if dof % 6 < 3)
    node, component = divmod(dof, 6)
    positions = state.positions.copy()
    positions[node, component] += .001
    invalid = replace(state, positions=positions)
    with pytest.raises(ValueError, match="fixed retained position changed"):
        context.assemble(invalid, .25, origin)
    with pytest.raises(ValueError, match="fixed retained position changed"):
        evaluate_local_trial(context, invalid, origin, .25,
                             family="generalized", element_index=0)
    with pytest.raises(ValueError, match="fixed retained position changed"):
        evaluate_sparse_trial(context, invalid, origin, .25,
                              family="generalized")


@pytest.mark.parametrize("case", (
    "generalized", "generalized_loaded", "physical_fibre"))
def test_typed_adapter_tangent_follows_native_state_update(case):
    if case.startswith("generalized"):
        fixture = build(4, z_offset=.0675,
                        torsion_constant=square_torsion_constant())
        context = fixture.context
        if case == "generalized_loaded":
            load = DistributedPattern(
                LinePattern(((1, .09, -.03, .02),)),
                ((1, .07, -.04, .05),))
            context = GeneralizedContext(
                fixture.model, GeneralizedProgram((1.,), load))
        layout = context
        advance = context.advance
    else:
        context = FibreContext(
            make_model(contrast=1.),
            ForceProgram((.25,), ((5, .4, -.005, 0.),),
                         max_iterations=24))
        layout = context.layout
        advance = layout.advance
    state, origin = context.initial.mechanical, context.initial.origins
    family = "physical_fibre" if case == "physical_fibre" else "generalized"
    direction = np.zeros(layout.count)
    first = layout.nodal_count-6
    direction[first:first+6] = (.4, -.3, .2, .15, -.1, .12)
    direction[layout.nodal_count:layout.nodal_count+7] = (
        .11, -.09, .07, .05, -.04, .03, .015)
    direction[np.asarray(layout.fixed, dtype=int)] = 0.
    state = advance(state, 1e-3*direction)
    base = evaluate_sparse_trial(context, state, origin, .25,
                                 family=family)
    assert np.linalg.norm(base.residual) > 1e-6
    expected = base.tangent@direction
    errors = []
    for step in (1e-5, 3e-6, 1e-6):
        plus = evaluate_sparse_trial(
            context, advance(state, step*direction), origin, .25,
            family=family).residual
        minus = evaluate_sparse_trial(
            context, advance(state, -step*direction), origin, .25,
            family=family).residual
        errors.append(_relative((plus-minus)/(2*step), expected))
    assert sum(error <= 1e-6 for error in errors) >= 2, errors
