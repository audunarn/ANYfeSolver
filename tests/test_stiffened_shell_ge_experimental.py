"""Integrated experimental Q4/native-B3 point-panel checks."""

import anyfesolver
from hashlib import sha256
import json
import numpy as np
import pytest
from dataclasses import replace
from scipy import sparse

from anyfesolver import b3_ge
from anyfesolver import _stiffened_shell_ge_experimental as ge_experimental
from anyfesolver._stiffened_shell_ge_checkpoint_binary import (
    MAGIC as BINARY_CHECKPOINT_MAGIC, decode as decode_binary_checkpoint,
    encode as encode_binary_checkpoint,
)
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver._stiffened_shell_ge_experimental import (
    PROFILE, ExperimentalGEAnalysis, FixedCoordinate, LinearDefinition,
    NodalAction, PointConnection, ShellPanel,
)


def _panel():
    e = 210e9
    nu = .3
    g = e/(2*(1+nu))
    section = b3_ge.EllipsoidalGeneralizedSection(
        np.diag((e*.01, g*.008, g*.008, g*1.6666666666666667e-5,
                 e*8.333333333333333e-6, e*8.333333333333333e-6)),
        np.eye(6), 1e20, 1e9)
    beam_points = [(x, -.5, .1) for x in (0., .5, 1., 1.5, 2.)]
    beams = tuple(b3_ge.define_beam(
        "b3-ge", i+1, (101+2*i, 102+2*i, 103+2*i),
        np.array(beam_points[2*i:2*i+3]), np.tile(np.eye(3), (3, 1, 1)),
        section, np.eye(6)) for i in range(2))
    shell = ShellPanel(
        (1, 2, 3, 4, 5, 6),
        ((0., -.5, 0.), (1., -.5, 0.), (2., -.5, 0.),
         (0., .5, 0.), (1., .5, 0.), (2., .5, 0.)),
        ((1, 2, 5, 4), (2, 3, 6, 5)), .07, e, nu, 7850.)
    return LinearDefinition(
        PROFILE, shell, beams,
        (PointConnection(1, 101), PointConnection(2, 103),
         PointConnection(3, 105)),
        tuple(FixedCoordinate("shell", node, axis)
              for node in (1, 4) for axis in range(6)))


def test_sparse_mixed_panel_solve_recovery_and_restart():
    definition = _panel()
    analysis = anyfesolver.stiffened_shell.create_analysis(definition=definition)
    assert analysis.capabilities["attachment"] == ("reference point pose",)
    legacy_identity = sha256(json.dumps({
        "profile": PROFILE, "shell_ids": definition.shell.node_ids,
        "shell_xyz": np.asarray(definition.shell.coordinates_m).tolist(),
        "shell_cells": definition.shell.q4_elements,
        "shell_s3_cells": definition.shell.s3_elements,
        "thickness": definition.shell.thickness_m,
        "E": definition.shell.young_pa, "nu": definition.shell.poisson,
        "density": definition.shell.density_kg_m3,
        "beams": [item.sha256 for item in definition.beams],
        "points": [(v.shell_node, v.beam_node) for v in definition.points],
        "fixed": [(v.domain, v.node, v.component) for v in definition.fixed],
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert analysis._identity == legacy_identity
    details = analysis.preflight()
    assert details["retained_dofs"] == 48
    assert details["constraints"] == 18
    assert details["production_qualified"] is False
    assert analysis._K.nnz < analysis._K.shape[0]**2
    actions = (NodalAction("beam", 105, (0., 0., 100., 0., 0., 0.)),)
    result = analysis.solve_linear(actions)
    assert result.status == "completed"
    assert result.free_equilibrium_ratio < 1e-8
    assert result.constraint_max < 1e-9
    assert result.beam_coordinates[6*4+2] > 0.
    assert np.linalg.norm(result.reactions[analysis._fixed]) > 0.
    support = result.reactions[:analysis._shell_count].reshape(-1, 6)
    support_force = support[:, :3].sum(axis=0)
    support_moment = (support[:, 3:].sum(axis=0) +
                      np.cross(np.asarray(definition.shell.coordinates_m),
                               support[:, :3]).sum(axis=0))
    np.testing.assert_allclose(support_force, (0., 0., -100.), atol=1e-6)
    np.testing.assert_allclose(support_moment, (50., 200., 0.), atol=1e-5)
    assert not result.shell_displacement.flags.writeable
    saved = analysis.checkpoint()
    with pytest.raises(ValueError):
        result.shell_displacement.setflags(write=True)
    assert analysis.checkpoint() == saved
    resumed = b3_ge.create_stiffened_shell_analysis(
        formulation="b3-ge", definition=definition)
    restored = resumed.restore(saved)
    np.testing.assert_allclose(restored.shell_displacement,
                               result.shell_displacement, rtol=0, atol=0)
    np.testing.assert_allclose(restored.beam_coordinates,
                               result.beam_coordinates, rtol=0, atol=0)
    second = resumed.solve_linear((NodalAction(
        "beam", 105, (0., 0., 200., 0., 0., 0.)),))
    np.testing.assert_allclose(second.beam_coordinates,
                               2*result.beam_coordinates, rtol=1e-7, atol=1e-12)
    assert second.generation == restored.generation+1
    corrupted = bytearray(saved)
    corrupted[-3] = ord('9')
    fresh = ExperimentalGEAnalysis(definition)
    with pytest.raises(ValueError):
        fresh.restore(bytes(corrupted))
    with pytest.raises(RuntimeError, match="fresh"):
        resumed.restore(saved)
    assert resumed.recover().generation == second.generation


def test_invalid_action_does_not_change_accepted_prefix():
    analysis = ExperimentalGEAnalysis(_panel())
    first = analysis.solve_linear((NodalAction(
        "beam", 105, (0., 0., 10., 0., 0., 0.)),))
    saved = analysis.checkpoint()
    with pytest.raises(ValueError, match="unknown"):
        analysis.solve_linear((NodalAction(
            "beam", 999, (0., 0., 1., 0., 0., 0.)),))
    assert analysis.recover().generation == first.generation
    assert analysis.checkpoint() == saved


def test_compiled_panel_owns_mutable_caller_geometry_and_connectivity():
    original = _panel()
    coordinates = [list(xyz) for xyz in original.shell.coordinates_m]
    cells = [list(cell) for cell in original.shell.q4_elements]
    shell = replace(original.shell, coordinates_m=coordinates, q4_elements=cells)
    caller_definition = replace(original, shell=shell)
    owned = anyfesolver.stiffened_shell.create_analysis(definition=caller_definition)
    control = anyfesolver.stiffened_shell.create_analysis(definition=original)
    identity = owned._identity

    # Mutation after compilation must not change the assembled operator,
    # recovery layout, or model identity used by a later checkpoint.
    coordinates[2][0] = 200.
    cells[1][1] = 1
    assert owned._identity == identity == control._identity
    actions = (NodalAction("beam", 105, (0., 0., 100., 0., 0., 0.)),)
    actual = owned.solve_linear(actions)
    expected = control.solve_linear(actions)
    np.testing.assert_array_equal(actual.shell_displacement,
                                  expected.shell_displacement)
    np.testing.assert_array_equal(actual.beam_coordinates,
                                  expected.beam_coordinates)
    np.testing.assert_array_equal(actual.reactions, expected.reactions)
    restored = ExperimentalGEAnalysis(original).restore(owned.checkpoint())
    np.testing.assert_array_equal(restored.shell_displacement,
                                  expected.shell_displacement)


def test_equivalent_shell_and_beam_renumbering_preserves_physical_result():
    original = _panel()
    shell_map = {1: 64, 2: 17, 3: 90, 4: 28, 5: 55, 6: 41}
    beam_map = {101: 330, 102: 415, 103: 205, 104: 580, 105: 125}
    order = (4, 1, 6, 2, 5, 3)
    original_xyz = dict(zip(original.shell.node_ids,
                            original.shell.coordinates_m))
    shell = replace(
        original.shell,
        node_ids=tuple(shell_map[node] for node in order),
        coordinates_m=tuple(original_xyz[node] for node in order),
        q4_elements=tuple(tuple(shell_map[node] for node in cell)
                          for cell in reversed(original.shell.q4_elements)))
    beams = []
    for index, definition in enumerate(reversed(original.beams)):
        element, inertia = definition.instantiate()
        reference = element.operator.reference
        beams.append(b3_ge.define_beam(
            "b3-ge", 70+index,
            tuple(beam_map[node] for node in element.node_ids),
            reference.coordinates, reference.nodal_triads,
            element.section, inertia))
    remapped = replace(
        original, shell=shell, beams=tuple(beams),
        points=tuple(PointConnection(shell_map[tie.shell_node],
                                     beam_map[tie.beam_node])
                     for tie in reversed(original.points)),
        fixed=tuple(FixedCoordinate("shell", shell_map[item.node],
                                    item.component)
                    for item in reversed(original.fixed)))
    baseline_owner = ExperimentalGEAnalysis(original)
    remapped_owner = ExperimentalGEAnalysis(remapped)
    baseline = baseline_owner.solve_linear((
        NodalAction("beam", 105, (0., 0., 100., 0., 0., 0.)),))
    candidate = remapped_owner.solve_linear((
        NodalAction("beam", beam_map[105], (0., 0., 100., 0., 0., 0.)),))

    shell_rows = baseline.shell_displacement.reshape(-1, 6)
    changed_shell_rows = candidate.shell_displacement.reshape(-1, 6)
    for node in original.shell.node_ids:
        np.testing.assert_allclose(
            shell_rows[baseline_owner._shell_index[node]],
            changed_shell_rows[remapped_owner._shell_index[shell_map[node]]],
            rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(
            baseline.reactions[:baseline_owner._shell_count].reshape(-1, 6)
            [baseline_owner._shell_index[node]],
            candidate.reactions[:remapped_owner._shell_count].reshape(-1, 6)
            [remapped_owner._shell_index[shell_map[node]]],
            rtol=1e-10, atol=1e-8)
    original_nodal = 6*len(baseline_owner._beam_ids)
    changed_nodal = 6*len(remapped_owner._beam_ids)
    for node in baseline_owner._beam_ids:
        np.testing.assert_allclose(
            baseline.beam_coordinates[
                6*baseline_owner._beam_index[node]:
                6*baseline_owner._beam_index[node]+6],
            candidate.beam_coordinates[
                6*remapped_owner._beam_index[beam_map[node]]:
                6*remapped_owner._beam_index[beam_map[node]]+6],
            rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        baseline.beam_coordinates[original_nodal:].reshape(-1, 24),
        candidate.beam_coordinates[changed_nodal:].reshape(-1, 24)[::-1],
        rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        baseline.multipliers.reshape(-1, 6),
        candidate.multipliers.reshape(-1, 6)[::-1],
        rtol=1e-10, atol=1e-12)
    assert baseline_owner.preflight()["identity"] != remapped_owner.preflight()["identity"]
    with pytest.raises(ValueError, match="incompatible"):
        ExperimentalGEAnalysis(remapped).restore(baseline_owner.checkpoint())
    restarted = ExperimentalGEAnalysis(remapped).restore(
        remapped_owner.checkpoint())
    np.testing.assert_array_equal(restarted.shell_displacement,
                                  candidate.shell_displacement)
    np.testing.assert_array_equal(restarted.beam_coordinates,
                                  candidate.beam_coordinates)


def test_reversed_native_member_parameter_preserves_point_panel_response():
    original = _panel()
    reversal = np.diag((-1., 1., -1.))
    reversed_beams = []
    for definition in reversed(original.beams):
        element, inertia = definition.instantiate()
        reference = element.operator.reference
        reversed_beams.append(b3_ge.define_beam(
            "b3-ge", element.element_id+30, element.node_ids[::-1],
            reference.coordinates[::-1],
            reference.nodal_triads[::-1]@reversal,
            element.section, inertia))
    changed = replace(original, beams=tuple(reversed_beams))
    initial = ExperimentalGEAnalysis(original)
    reversed_owner = ExperimentalGEAnalysis(changed)
    for wrench in ((0., 0., 100., 0., 0., 0.),
                   (0., 100., 0., 0., 0., 0.),
                   (0., 0., 0., 10., 0., 0.)):
        action = (NodalAction("beam", 105, wrench),)
        baseline = initial.solve_linear(action)
        candidate = reversed_owner.solve_linear(action)
        np.testing.assert_allclose(candidate.shell_displacement,
                                   baseline.shell_displacement,
                                   rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(candidate.beam_coordinates[:30],
                                   baseline.beam_coordinates[:30],
                                   rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(candidate.reactions[:initial._shell_count],
                                   baseline.reactions[:initial._shell_count],
                                   rtol=1e-10, atol=1e-8)


def test_reference_linear_load_cases_reuse_one_factorization():
    analysis = ExperimentalGEAnalysis(_panel())
    assert analysis.preflight()["reference_linear_factorizations"] == 0
    first = analysis.solve_linear((NodalAction(
        "beam", 105, (0., 0., 100., 0., 0., 0.)),))
    second = analysis.solve_linear((NodalAction(
        "beam", 105, (0., 0., 200., 0., 0., 0.)),))
    np.testing.assert_allclose(second.beam_coordinates,
                               2*first.beam_coordinates, rtol=1e-10, atol=1e-12)
    assert analysis.preflight()["reference_linear_factorizations"] == 1
    assert analysis.preflight()["reference_linear_refinements"] == 4
    assert second.generation == first.generation+1


@pytest.mark.parametrize("correction_error", [0., .2])
def test_linear_refinement_recovers_a_roundoff_sized_response_error(monkeypatch, correction_error):
    analysis = ExperimentalGEAnalysis(_panel())
    factor = ge_experimental.spla.splu(analysis._A)
    calls = []

    class RoundedFactor:
        def solve(self, rhs):
            answer = factor.solve(rhs)
            calls.append(np.array(rhs, copy=True))
            # This relative defect fits inside the physical equilibrium gate,
            # but is too large for equivalent-response precision.
            if len(calls) == 1:
                return answer*(1.+3.e-9)
            # A rounded first correction can leave a forward error below the
            # equilibrium gate but above equivalent-response precision.
            return answer*(1.+correction_error) if len(calls) == 2 else answer

    monkeypatch.setattr(ge_experimental.spla, "splu",
                        lambda _matrix: RoundedFactor())
    observed = analysis.solve_linear((NodalAction(
        "beam", 105, (0., 0., 100., 0., 0., 0.)),))
    dense = np.linalg.solve(analysis._A.toarray(), calls[0])
    actual = np.r_[np.r_[observed.shell_displacement,
                        observed.beam_coordinates][analysis._free],
                   observed.multipliers]
    assert np.linalg.norm(actual-dense)/np.linalg.norm(dense) <= 1.e-10
    assert len(calls) == 3
    assert analysis.preflight()["reference_linear_refinements"] == 2


@pytest.mark.parametrize("failure", ["cancel", "cancel_second", "numerical", "programming"])
def test_linear_refinement_failure_preserves_accepted_snapshot(monkeypatch, failure):
    from anyfesolver import _b2_physical_linear_batch_dev as residual_service
    analysis = ExperimentalGEAnalysis(_panel())
    action = (NodalAction("beam", 105, (0., 0., 100., 0., 0., 0.)),)
    accepted = analysis.solve_linear(action)
    raw = analysis.checkpoint()
    token = CancellationToken()
    if failure in ("cancel", "cancel_second"):
        safe_point = ge_experimental.cancellation_safe_point
        refinements = 0

        def cancel_refinement(current_token, phase):
            nonlocal refinements
            if phase == "stiffened-shell.linear.refined":
                refinements += 1
                if refinements == (2 if failure == "cancel_second" else 1):
                    current_token.cancel("after accurate refinement")
            return safe_point(current_token, phase)

        monkeypatch.setattr(ge_experimental, "cancellation_safe_point", cancel_refinement)
        expected, message = SolveCancelled, "after accurate refinement"
    else:
        error = (residual_service.B2NumericalSolveFailure("unrepresentable")
                 if failure == "numerical" else ValueError("fixture programming error"))

        def failed_residual(*args):
            raise error

        monkeypatch.setattr(residual_service, "_accurate_sparse_residual", failed_residual)
        expected = RuntimeError if failure == "numerical" else ValueError
        message = "residual is not representable" if failure == "numerical" else "fixture programming"
    with pytest.raises(expected, match=message) as caught:
        analysis.solve_linear(action, cancellation_token=token)
    if failure == "numerical":
        assert isinstance(caught.value.__cause__, residual_service.B2NumericalSolveFailure)
    assert analysis.checkpoint() == raw
    assert analysis._generation == accepted.generation
    assert analysis.preflight()["reference_linear_factorizations"] == 1


def test_binary_checkpoint_restores_linear_state_and_rejects_corruption(monkeypatch):
    analysis = ExperimentalGEAnalysis(_panel())
    accepted = analysis.solve_linear((NodalAction(
        "beam", 105, (0., 0., 100., 0., 0., 0.)),))
    monkeypatch.setattr(ge_experimental, "MAX_JSON_CHECKPOINT_VALUES", 0)
    raw = analysis.checkpoint()
    assert raw.startswith(BINARY_CHECKPOINT_MAGIC)
    restored = ExperimentalGEAnalysis(_panel()).restore(raw)
    np.testing.assert_array_equal(restored.shell_displacement,
                                  accepted.shell_displacement)
    np.testing.assert_array_equal(restored.beam_coordinates,
                                  accepted.beam_coordinates)
    np.testing.assert_array_equal(restored.reactions, accepted.reactions)
    damaged = bytearray(raw)
    damaged[-35] ^= 1
    fresh = ExperimentalGEAnalysis(_panel())
    with pytest.raises(ValueError, match="digest"):
        fresh.restore(bytes(damaged))
    assert fresh._accepted is None and fresh._generation == 0
    nonphysical_q = np.r_[accepted.shell_displacement,
                          accepted.beam_coordinates].copy()
    nonphysical_q[0] = 1.e-3  # fixed support, despite a valid byte digest
    forged = encode_binary_checkpoint(
        "linear", analysis._identity, accepted.generation,
        (("q", nonphysical_q), ("multipliers", accepted.multipliers),
         ("load", analysis._accepted[1])))
    with pytest.raises(ValueError, match="mechanically inconsistent"):
        fresh.restore(forged)
    assert fresh._accepted is None and fresh._generation == 0
    huge_load = np.zeros(analysis._K.shape[0])
    huge_load[analysis._free[0]] = 1.e308
    overflow_norm = encode_binary_checkpoint(
        "linear", analysis._identity, 1,
        (("q", np.zeros_like(nonphysical_q)),
         ("multipliers", np.zeros_like(accepted.multipliers)),
         ("load", huge_load)))
    with pytest.raises(ValueError, match="mechanically inconsistent"):
        fresh.restore(overflow_norm)
    assert fresh._accepted is None and fresh._generation == 0


def test_binary_checkpoint_supports_model_sized_arrays_beyond_json_limit():
    values = np.arange(300_000, dtype=float) / 16.
    raw = encode_binary_checkpoint("linear", "a" * 64, 1,
                                   (("q", values), ("multipliers", np.zeros(2)),
                                    ("load", np.zeros(3))))
    assert len(raw) > 2_000_000
    kind, generation, recovered = decode_binary_checkpoint(
        raw, "a" * 64,
        lambda _kind, _generation: (("q", (300_000,)),
                                    ("multipliers", (2,)), ("load", (3,))))
    assert (kind, generation) == ("linear", 1)
    np.testing.assert_array_equal(recovered["q"], values)
    with pytest.raises(ValueError, match="incompatible"):
        decode_binary_checkpoint(raw, "b" * 64, lambda *_: ())


def test_stress_free_reference_check_is_rowwise_and_unit_covariant():
    tangent = np.array(((2., -3., 0.), (0., 4., 5.), (0., 0., 0.)))
    residual = np.array((1.e-7, -2.e-7, 0.))
    coordinate_scale = np.array((3., 2., 4.))
    original = ge_experimental._scaled_reference_residual(
        residual, tangent, coordinate_scale)
    units = np.diag((1.e3, 1.e-3, 1.e6))
    converted = ge_experimental._scaled_reference_residual(
        units@residual, units@tangent@units,
        np.linalg.solve(units, coordinate_scale))
    np.testing.assert_allclose(converted, original, rtol=1.e-14)
    assert np.isinf(ge_experimental._scaled_reference_residual(
        np.array((0., 0., 1.)), tangent, coordinate_scale))


def test_mixed_equilibrium_and_constraint_checks_are_work_covariant():
    residual = np.array((.003, -.02, 1.e-4))
    applied = np.array((2., -3., .5))
    coordinate_scale = np.array((.2, 1., 4.))
    original = ge_experimental._relative_equilibrium(
        residual, applied, coordinate_scale, .2)
    units = np.diag((1.e3, 1.e-3, 1.e6))
    converted = ge_experimental._relative_equilibrium(
        units@residual, units@applied,
        np.linalg.solve(units, coordinate_scale), .2)
    np.testing.assert_allclose(converted, original, rtol=1.e-14)
    linear = ge_experimental._linear_equilibrium(
        residual, applied, coordinate_scale)
    converted_linear = ge_experimental._linear_equilibrium(
        units@residual, units@applied,
        np.linalg.solve(units, coordinate_scale))
    np.testing.assert_allclose(converted_linear, linear, rtol=1.e-14)

    jacobian = np.array(((1., -2., 0.), (0., 3., .5)))
    gap = np.array((.003, -.02))
    original_gap = ge_experimental._scaled_constraint_gap(
        gap, np.abs(jacobian)@coordinate_scale)
    row_units = np.diag((1.e4, 1.e-5))
    transformed_jacobian = row_units@jacobian@units
    converted_gap = ge_experimental._scaled_constraint_gap(
        row_units@gap,
        np.abs(transformed_jacobian)@
        np.linalg.solve(units, coordinate_scale))
    np.testing.assert_allclose(converted_gap, original_gap, rtol=1.e-14)


def test_reference_linear_tiny_load_requires_an_actual_response(monkeypatch):
    definition = _panel()
    tiny = (NodalAction("beam", 105, (1e-12, 0., 0., 0., 0., 0.)),)
    normal = (NodalAction("beam", 105, (1., 0., 0., 0., 0., 0.)),)
    reference = ExperimentalGEAnalysis(definition).solve_linear(normal)
    tiny_owner = ExperimentalGEAnalysis(definition)
    actual = tiny_owner.solve_linear(tiny)
    np.testing.assert_allclose(actual.beam_coordinates,
                               1e-12*reference.beam_coordinates,
                               rtol=1e-6, atol=1e-22)
    restored = ExperimentalGEAnalysis(definition).restore(tiny_owner.checkpoint())
    np.testing.assert_allclose(restored.beam_coordinates,
                               actual.beam_coordinates, rtol=0., atol=0.)

    class ZeroFactor:
        def solve(self, rhs):
            return np.zeros_like(rhs)

    monkeypatch.setattr(ge_experimental.spla, "splu", lambda _matrix: ZeroFactor())
    failed = ExperimentalGEAnalysis(definition)
    with pytest.raises(RuntimeError, match="mixed reference linear equilibrium"):
        failed.solve_linear(tiny)
    assert failed._generation == 0 and failed._accepted is None


def test_reference_linear_large_load_cannot_hide_a_missed_small_component(
        monkeypatch):
    analysis = ExperimentalGEAnalysis(_panel())
    tip_y = analysis._shell_count+6*analysis._beam_index[105]+1
    reduced_y = int(np.flatnonzero(analysis._free == tip_y)[0])
    actual_factor = ge_experimental.spla.splu(analysis._A)

    class DropSmallComponent:
        def solve(self, rhs):
            omitted = np.array(rhs, copy=True)
            omitted[reduced_y] = 0.
            return actual_factor.solve(omitted)

    monkeypatch.setattr(ge_experimental.spla, "splu",
                        lambda _matrix: DropSmallComponent())
    mixed = (NodalAction("beam", 105, (1., 1e-12, 0., 0., 0., 0.)),)
    with pytest.raises(RuntimeError, match="mixed reference linear equilibrium"):
        analysis.solve_linear(mixed)
    assert analysis._generation == 0 and analysis._accepted is None


def test_compiled_stress_free_reference_reports_physical_scale():
    details = ExperimentalGEAnalysis(_panel()).preflight()
    assert details["reference_length_m"] > 0.
    assert details["stress_free_scaled_max"] <= 32.*np.finfo(float).eps
    assert details["equilibrium_norm"] == "frozen_work_weighted_l2_plus_active_load_rows"
    assert details["constraint_norm"] == "reference_row_scaled_max"


def test_cancellation_after_linear_factorization_preserves_empty_prefix(monkeypatch):
    analysis = ExperimentalGEAnalysis(_panel())
    action = (NodalAction("beam", 105, (0., 0., 100., 0., 0., 0.)),)
    token = CancellationToken()
    safe_point = ge_experimental.cancellation_safe_point

    def cancel_after_solve(current_token, phase):
        if phase == "stiffened-shell.linear.solved":
            current_token.cancel("after sparse solve")
        return safe_point(current_token, phase)

    monkeypatch.setattr(ge_experimental, "cancellation_safe_point",
                        cancel_after_solve)
    with pytest.raises(SolveCancelled, match="after sparse solve"):
        analysis.solve_linear(action, cancellation_token=token)
    assert analysis._generation == 0 and analysis._accepted is None
    assert analysis.preflight()["reference_linear_factorizations"] == 1
    monkeypatch.setattr(ge_experimental, "cancellation_safe_point", safe_point)
    accepted = analysis.solve_linear(action)
    assert accepted.generation == 1
    assert analysis.preflight()["reference_linear_factorizations"] == 1


@pytest.mark.parametrize("q4,s3", [
    (((1, 2, 5, 4),), ((2, 3, 6), (2, 6, 5))),
    ((), ((1, 2, 5), (1, 5, 4), (2, 3, 6), (2, 6, 5))),
])
def test_mixed_and_s3_point_panel_sparse_recovery_restart(q4, s3):
    original = _panel()
    definition = replace(original, shell=replace(
        original.shell, q4_elements=q4, s3_elements=s3))
    analysis = ExperimentalGEAnalysis(definition)
    assert analysis.preflight()["retained_dofs"] == 48
    actions = (NodalAction("beam", 105, (0., 0., 100., 0., 0., 0.)),)
    first = analysis.solve_linear(actions)
    assert first.free_equilibrium_ratio < 1e-8
    assert first.constraint_max < 1e-9
    support = first.reactions.reshape(-1, 6)
    np.testing.assert_allclose(support[:, :3].sum(axis=0),
                               (0., 0., -100.), atol=1e-5)
    restored = ExperimentalGEAnalysis(definition).restore(analysis.checkpoint())
    np.testing.assert_array_equal(restored.shell_displacement,
                                  first.shell_displacement)
    np.testing.assert_array_equal(restored.beam_coordinates,
                                  first.beam_coordinates)


def test_disconnected_shell_cells_fail_before_sparse_solve():
    original = _panel()
    shell = replace(original.shell, q4_elements=((1, 2, 5, 4), (3, 7, 8, 6)),
                    node_ids=original.shell.node_ids+(7, 8),
                    coordinates_m=original.shell.coordinates_m+
                    ((3., -.5, 0.), (3., .5, 0.)))
    with pytest.raises(ValueError, match="connected"):
        ExperimentalGEAnalysis(replace(original, shell=shell))


def test_duplicate_point_equations_fail_during_preflight():
    definition = _panel()
    with pytest.raises(ValueError, match="duplicate experimental point"):
        ExperimentalGEAnalysis(replace(
            definition, points=definition.points+(definition.points[0],)))


def test_point_constraint_supported_on_both_sides_fails_preflight():
    definition = _panel()
    additional = tuple(FixedCoordinate(domain, node, axis)
                       for domain, node in (("shell", 2), ("beam", 103))
                       for axis in range(6))
    with pytest.raises(ValueError, match="point shell=2 beam=103"):
        ExperimentalGEAnalysis(replace(
            definition, fixed=definition.fixed+additional))


def test_large_attachment_rank_screen_stays_sparse(monkeypatch):
    rows = sparse.eye(60, 100, format="csr")
    free = np.arange(100)
    scales = np.ones(100)
    row_scales = np.ones(60)

    def forbid_dense(_self):
        raise AssertionError("large attachment audit materialized dense rows")

    monkeypatch.setattr(sparse.csr_matrix, "toarray", forbid_dense)
    ge_experimental._require_independent_supported_rows(
        rows, free, scales, row_scales)
    dependent = rows.tolil()
    dependent[59, :] = rows[0, :]
    with pytest.raises(ValueError, match="structurally independent"):
        ge_experimental._require_independent_supported_rows(
            dependent.tocsr(), free, scales, row_scales)


def test_sparse_solution_matches_bounded_dense_kkt_and_cancel_keeps_prefix():
    analysis = ExperimentalGEAnalysis(_panel())
    action = (NodalAction("beam", 105, (0., 0., 100., 0., 0., 0.)),)
    observed = analysis.solve_linear(action)
    load = np.zeros(analysis._K.shape[0])
    load[analysis._shell_count+6*analysis._beam_index[105]+2] = 100.
    dense = np.linalg.solve(analysis._A.toarray(),
                            np.r_[load[analysis._free],
                                  np.zeros(analysis._C.shape[0])])
    np.testing.assert_allclose(
        np.r_[observed.shell_displacement, observed.beam_coordinates]
        [analysis._free], dense[:len(analysis._free)], rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(observed.multipliers,
                               dense[len(analysis._free):], rtol=1e-9, atol=1e-9)
    saved = analysis.checkpoint()
    token = CancellationToken()
    token.cancel("fixture cancellation")
    with pytest.raises(SolveCancelled, match="fixture cancellation"):
        analysis.solve_linear(action, cancellation_token=token)
    assert analysis.checkpoint() == saved
