"""Focused correctness tests for the opt-in prepared static evaluator."""
from __future__ import annotations

import gc

import numpy as np
import pytest

from anyfesolver import (
    AnalysisSession,
    BoundaryCondition,
    CancellationToken,
    FixedSupport,
    LoadCase,
    PreparedStaticCapabilityError,
    PreparedStaticError,
    RecoveryConfig,
    ResourceConfig,
    SolveCancelled,
    StaticCase,
    compute_constraint_force_diagnostics,
    prepare_static_evaluator,
    solve_linear,
)
from anyfesolver.e4_pl_s3_element import QualifiedE4PLS3ShellElement
from anyfesolver.elements import BeamElement
from anyfesolver.fe_core import FEModel
from anyfesolver.recovery import ResourcePolicyError, recover_stress_result


def _bar(supported: bool = True):
    model = FEModel("prepared-static-test")
    model.add_material("steel", 100.0, 0.3, density=2.0)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_element(1, BeamElement(1, [1, 2], "steel",
                                     {"area": 1.0, "Iy": 1e-3, "Iz": 1e-3, "J": 1e-3}))
    if supported:
        model.add_boundary_condition(FixedSupport("fixed", [1]))
        model.add_boundary_condition(BoundaryCondition("slide", [2],
            {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}))
    return model


def _case(index, force=1.0, balanced=False):
    load = LoadCase(str(index))
    load.add_nodal_load(2, [force, 0.0, 0.0, 0.0, 0.0, 0.0])
    if balanced:
        load.add_nodal_load(1, [-force, 0.0, 0.0, 0.0, 0.0, 0.0])
    return StaticCase(str(index), load, float(index) ** 2)


def _shell_model():
    model = FEModel("prepared-static-shell")
    model.add_material("steel", 210.0e9, 0.3, density=100.0)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_node(3, 0.0, 1.0, 0.0)
    model.add_node(4, 1.0, 1.0, 0.0)
    model.add_element(1, QualifiedE4PLS3ShellElement(
        1, [1, 2, 3], "steel", thickness=0.01, reference_normal=(0.0, 0.0, 1.0)))
    model.add_element(2, QualifiedE4PLS3ShellElement(
        2, [2, 4, 3], "steel", thickness=0.01, reference_normal=(0.0, 0.0, 1.0)))
    model.add_boundary_condition(FixedSupport("fixed", [1, 3]))
    return model


def _shell_case(index, *, pressure=0.0, gravity=False, added=0.0, follower=False):
    load = LoadCase(f"shell-{index}")
    load.add_nodal_load(4, [0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
    if pressure:
        load.pressure_loads[1] = pressure
        load.pressure_loads[2] = pressure
        load.follower_pressure = follower
    if gravity:
        load.set_gravity(0.0, 0.0, -9.81)
    if added:
        load.add_node_mass(4, added)
    return StaticCase(f"shell-{index}", load, float(index))


def test_supported_beam_parity_and_certification():
    model = _bar()
    cases = [_case(i, force) for i, force in enumerate([1.0, -3.0, 0.2, 5.0, -1.0])]
    expected = [solve_linear(model, case.load_case)[0] for case in cases]
    with prepare_static_evaluator(model, batch_size=2) as evaluator:
        blocks = list(evaluator.iter_batches(cases))
        assert [len(block.case_ids) for block in blocks] == [2, 2, 1]
        actual = np.column_stack([block.displacements for block in blocks])
        np.testing.assert_allclose(actual, np.column_stack(expected), rtol=1e-12, atol=1e-14)
        for block in blocks:
            assert block.times == tuple(float(case.time) for case in
                                        cases[block.input_indices[0]:block.input_indices[-1] + 1])
            for item in block.case_results:
                info = item["solver_info"]
                assert info["status"] == "converged"
                assert info["relative_backward_error_disposition"] == "CERTIFIED"
                assert info["relative_backward_error"] <= 1e-10
                assert info["relative_backward_error_limit"] == 1e-10
        diagnostics = evaluator.diagnostics()
        assert diagnostics["cases_certified"] == 5
        assert diagnostics["blocks_yielded"] == 3
        assert diagnostics["factorization_cache"]["misses"] == 1
        assert diagnostics["factorization_cache"]["name"] == "prepared_static_evaluator"
        assert not diagnostics["closed"]


def test_snapshot_survives_mutation_deletion_and_external_factorization():
    model = _bar()
    cases = [_case(0, 2.0), _case(1, -1.0)]
    expected = [solve_linear(model, case.load_case)[0] for case in cases]
    with AnalysisSession(model) as session:
        evaluator = prepare_static_evaluator(model)
        # An external same-pattern factorization through another cache must
        # not disturb the evaluator's private factor.
        session.static_operator_plan(model=model)
    session.close()
    assert session.closed
    model.mesh.nodes[2].x = 7.0
    model.mesh.elements[1].material_name = "mutated"
    del model
    gc.collect()
    blocks = list(evaluator.iter_batches(cases))
    actual = np.column_stack([block.displacements for block in blocks])
    np.testing.assert_allclose(actual, np.column_stack(expected), rtol=1e-12, atol=1e-14)
    evaluator.close()
    assert evaluator.closed
    with pytest.raises(RuntimeError):
        evaluator.iter_batches(cases)


def test_free_free_balanced_gauge_and_incompatible_refusal():
    model = _bar(False)
    cases = [_case(0, 1.0, True), _case(1, -2.0, True)]
    expected = [solve_linear(model, case.load_case)[0] for case in cases]
    with prepare_static_evaluator(model, batch_size=1) as evaluator:
        blocks = list(evaluator.iter_batches(cases))
        for j, block in enumerate(blocks):
            np.testing.assert_allclose(block.displacements[:, 0], expected[j],
                                       rtol=1e-11, atol=1e-13)
            info = block.case_results[0]["solver_info"]
            assert info["method"] == "rigid_body_nullspace_augmented"
            assert info["gauge_residual_norm"] < 1e-12
            assert len(info["rigid_body_lagrange_multipliers"]) == \
                info["nullspace_rank"]
        assert evaluator.diagnostics()["augmented"]

    incompatible = _case(0)
    with prepare_static_evaluator(model) as evaluator:
        with pytest.raises(PreparedStaticError) as error:
            list(evaluator.iter_batches([incompatible]))
        assert error.value.diagnostics["status"] == "incompatible_free_free_load"
        assert error.value.diagnostics["case_id"] == "0"
    expected_gauged, _ = solve_linear(model, incompatible.load_case,
                                      allow_unbalanced_free_free=True)
    with prepare_static_evaluator(model, allow_unbalanced_free_free=True) as evaluator:
        block = next(evaluator.iter_batches([incompatible]))
        np.testing.assert_allclose(block.displacements[:, 0], expected_gauged,
                                   rtol=1e-11, atol=1e-13)
        assert block.case_results[0]["solver_info"]["warnings"]


def test_shell_pressure_gravity_added_mass_and_reaction_parity():
    model = _shell_model()
    cases = [
        _shell_case(0, pressure=5.0),
        _shell_case(1, pressure=-2.0, gravity=True, added=3.0),
        _shell_case(2, pressure=1.0, follower=True),
    ]
    expected = [solve_linear(model, case.load_case)[0] for case in cases]
    expected_forces = [compute_constraint_force_diagnostics(model, u, case.load_case)
                      for u, case in zip(expected, cases)]
    recovery = RecoveryConfig(include_displacements=False, include_stresses=True,
                              include_reactions=True)
    with prepare_static_evaluator(model, recovery_config=recovery, batch_size=2) as evaluator:
        blocks = list(evaluator.iter_batches(cases))
        actual = np.column_stack([block.displacements for block in blocks])
        np.testing.assert_allclose(actual, np.column_stack(expected), rtol=1e-11, atol=1e-13)
        flat = [item for block in blocks for item in block.case_results]
        for j, item in enumerate(flat):
            forces = item["constraint_forces"]
            for key in ("support_reactions", "mpc_slave_forces",
                        "mpc_master_equivalent_forces"):
                assert set(forces[key]) == set(expected_forces[j][key]), (j, key)
                for node in expected_forces[j][key]:
                    np.testing.assert_allclose(forces[key][node],
                                                expected_forces[j][key][node])
            assert item["reactions"].keys() >= forces["support_reactions"].keys()
        diagnostics = evaluator.diagnostics()
        verification = diagnostics["snapshot"]["load_map_verification"]
        assert verification["nodal"] <= 1e-11
        assert verification["pressure"] <= 1e-11
        assert verification["acceleration"] <= 1e-11
        assert diagnostics["snapshot"]["recovery_compiled_elements"] == 2


def test_compiled_stress_recovery_matches_live_oracle():
    model = _shell_model()
    recovery = RecoveryConfig(include_displacements=False, include_stresses=True,
                             include_reactions=False)
    rng = np.random.default_rng(20261008)
    total_dofs = model.mesh.dof_manager.total_dofs
    displacement = rng.uniform(-1e-4, 1e-4, total_dofs)
    oracle = recover_stress_result(model, displacement, recovery)
    with prepare_static_evaluator(model, recovery_config=recovery) as evaluator:
        block = next(evaluator.iter_batches([StaticCase("probe", None, 0.0)]))
        # The compiled maps are exercised through the public route below; the
        # zero-load block only certifies the solve.  Apply the compiled maps
        # through a second evaluator call with the displacement injected via
        # the affine structure: recovery of u is baseline + A @ u, verified
        # directly against the oracle here.
        from anyfesolver.prepared_static_recovery import apply_compiled_recovery
        compilation = evaluator.diagnostics()
        assert compilation["snapshot"]["recovery_compiled_elements"] == 2
        snapshot = evaluator._snapshot
        stresses, _provenance, _execution = apply_compiled_recovery(
            snapshot.recovery_compilation, displacement)
        for element_id, expected in oracle.element_stresses.items():
            compiled = stresses[int(element_id)]
            assert set(compiled) == set(expected)
            for name, values in expected.items():
                if np.asarray(values).dtype.kind not in "biufc":
                    assert compiled[name] == values
                    continue
                np.testing.assert_allclose(np.asarray(compiled[name], dtype=float),
                                           np.asarray(values, dtype=float),
                                           rtol=1e-9, atol=1e-15)
        # And through the public route: solve a case, then compare the block's
        # recovered stresses with a live recovery of the same solved field.
        case = _shell_case(0, pressure=3.0)
        solved, _info = solve_linear(model, case.load_case)
        live = recover_stress_result(model, solved, recovery)
        block = next(evaluator.iter_batches([case]))
        recovered = block.case_results[0]["element_stresses"]
        for element_id, values in live.element_stresses.items():
            for name, expected_values in values.items():
                if np.asarray(expected_values).dtype.kind not in "biufc":
                    assert recovered[int(element_id)][name] == expected_values
                    continue
                np.testing.assert_allclose(
                    np.asarray(recovered[int(element_id)][name], dtype=float),
                    np.asarray(expected_values, dtype=float),
                    rtol=1e-9, atol=1e-15)


def test_foreign_load_case_and_family_restrictions_are_refused():
    model = _bar()

    class ForeignLoadCase(LoadCase):
        pass

    with prepare_static_evaluator(model) as evaluator:
        with pytest.raises(PreparedStaticCapabilityError) as error:
            list(evaluator.iter_batches([StaticCase("f", ForeignLoadCase("x"), None)]))
        assert error.value.diagnostics["case_id"] == "f"

    with pytest.raises(ValueError):
        prepare_static_evaluator(model, load_families=["nodal", "magnetic"])
    with prepare_static_evaluator(model, load_families=["nodal"]) as evaluator:
        load = LoadCase("g")
        load.set_gravity(0.0, 0.0, -9.81)
        with pytest.raises(PreparedStaticCapabilityError) as error:
            list(evaluator.iter_batches([StaticCase("g", load, None)]))
        assert error.value.diagnostics["family"] == "acceleration"


def test_unadmitted_stress_family_is_refused_at_preparation():
    from anyfesolver.materials import Hill48Yield

    model = _shell_model()
    model.add_material("hill", 210.0e9, 0.3, density=100.0)
    model.materials["hill"].hill_yield = Hill48Yield(
        400.0e6, 350.0e6, 300.0e6, 220.0e6, 210.0e6, 200.0e6)
    model.add_element(3, QualifiedE4PLS3ShellElement(
        3, [1, 2, 3], "hill", thickness=0.01, reference_normal=(0.0, 0.0, 1.0)))
    recovery = RecoveryConfig(include_displacements=False, include_stresses=True,
                               include_reactions=False)
    with pytest.raises(PreparedStaticCapabilityError) as error:
        prepare_static_evaluator(model, recovery_config=recovery)
    refused = error.value.diagnostics["refused"]
    assert any(record["element_id"] == 3 for record in refused)
    # Restricting the selection to admitted elements prepares successfully.
    limited = RecoveryConfig(element_ids=[1, 2], include_displacements=False,
                             include_stresses=True, include_reactions=False)
    with prepare_static_evaluator(model, recovery_config=limited) as evaluator:
        assert evaluator.diagnostics()["snapshot"]["recovery_compiled_elements"] == 2


def test_cancellation_progress_and_case_labelled_failures():
    model = _bar()
    token = CancellationToken()
    token.cancel("stop")
    with prepare_static_evaluator(model) as evaluator:
        with pytest.raises(SolveCancelled):
            list(evaluator.iter_batches([_case(0)], cancellation_token=token))

    events = []
    with prepare_static_evaluator(model) as evaluator:
        blocks = list(evaluator.iter_batches(
            [_case(0), _case(1)], progress_callback=events.append))
        assert len(blocks) == 1
        assert events and events[-1]["type"] == "prepared_static_batch_complete"
        assert events[-1]["completed"] == 2

    bad = LoadCase("bad")
    bad.add_nodal_load(2, [float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0])
    with prepare_static_evaluator(model) as evaluator:
        with pytest.raises(PreparedStaticError) as error:
            list(evaluator.iter_batches([_case(0), StaticCase("bad", bad, None)]))
        diagnostics = error.value.diagnostics
        assert diagnostics["case_id"] == "bad"
        assert diagnostics["input_index"] == 1


def test_memory_limit_refuses_oversized_case_before_solve():
    model = _bar()
    cases = [_case(i, 1.0) for i in range(3)]
    tiny = ResourceConfig(memory_limit_bytes=1024)
    with pytest.raises(ResourcePolicyError):
        prepare_static_evaluator(model, resource_config=tiny)


def test_default_batch_size_and_effective_reduction():
    model = _bar()
    cases = [_case(i, 1.0) for i in range(40)]
    with prepare_static_evaluator(model) as evaluator:
        blocks = list(evaluator.iter_batches(cases))
        assert [len(block.case_ids) for block in blocks] == [32, 8]
        assert blocks[0].diagnostics["requested_batch_size"] == 32
        assert blocks[0].diagnostics["effective_batch_size"] == 32
    limit = ResourceConfig(memory_limit_bytes=20_000_000)
    with prepare_static_evaluator(model, resource_config=limit, batch_size=32) as evaluator:
        blocks = list(evaluator.iter_batches(cases))
        assert blocks[0].diagnostics["requested_batch_size"] == 32
        assert blocks[0].diagnostics["effective_batch_size"] <= 32
        assert sum(len(block.case_ids) for block in blocks) == 40
