"""V01 development parity for equivalent legacy and sparse one-shell models."""

import numpy as np
import pytest
from scipy import sparse

from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition
from anyfesolver._stiffened_shell_ge_experimental import (
    PROFILE, ExperimentalGEAnalysis, FixedCoordinate, LinearDefinition,
    NodalAction, PointConnection, ShellPanel,
)

from test_ge_beam3_coupled_modes import fixture as legacy_fixture
from test_ge_beam3_coupled_free_beam import build as legacy_direct_fixture
from test_ge_beam3_native_generalized_modal import inertia


def _matching_owners(topology, *, direct_q4=False):
    legacy = (legacy_direct_fixture(topology, False) if direct_q4
              else legacy_fixture(topology, False))
    assembly = legacy.assembly
    node_ids = tuple(range(1, len(assembly.coordinates)+1))
    panel = ShellPanel(
        node_ids, tuple(map(tuple, assembly.coordinates)),
        ((1, 2, 3, 4),) if topology == "Q4" else (),
        .2, 1000., .3, 2.,
        ((1, 2, 3),) if topology == "S3-V2D" else ())
    beams = tuple(NativeBeamDefinition.capture(element, inertia())
                  for _eid, element in assembly.beam.elements)
    supports = tuple(FixedCoordinate("shell", row//6+1, row%6)
                     for row in legacy.shell_fixed)
    current = ExperimentalGEAnalysis(LinearDefinition(
        PROFILE, panel, beams,
        (PointConnection(assembly.shell_node+1, assembly.beam_node),),
        supports))
    return legacy, current


@pytest.mark.parametrize("topology", ("Q4", "S3-V2D"))
def test_matching_one_shell_reference_tangent_solution_and_reactions(topology):
    legacy, current = _matching_owners(topology)
    assembly = legacy.assembly

    state = legacy.initial
    response, old_residual, _, _ = legacy._evaluate(
        state, state.mechanical, state.shell_u, state.multipliers, 0)
    full = sparse.bmat([[current._K, current._C.T],
                        [current._C, None]], format="csr").toarray()
    assert full.shape == response.tangent.shape
    assert np.max(np.abs(old_residual)) == 0.
    assert current.preflight()["stress_free_scaled_max"] < 1e-12
    tangent_error = np.linalg.norm(full-response.tangent) / np.linalg.norm(
        response.tangent)
    assert tangent_error <= 1e-10

    load = np.zeros(assembly.count)
    load[:len(legacy.forces)] = legacy.forces
    old_solution = np.zeros(assembly.count)
    old_solution[list(legacy.free)] = np.linalg.solve(
        response.tangent[np.ix_(legacy.free, legacy.free)],
        load[list(legacy.free)])
    result = current.solve_linear((
        NodalAction("beam", 3, (0., .0001, .0002, 0., 0., 0.)),))
    new_solution = np.r_[result.shell_displacement,
                         result.beam_coordinates, result.multipliers]
    np.testing.assert_array_equal(current._accepted[1],
                                  load[:current._K.shape[0]])
    solution_error = np.linalg.norm(new_solution-old_solution) / np.linalg.norm(
        old_solution)
    assert solution_error <= 1e-10
    old_reactions = response.tangent@old_solution-load
    np.testing.assert_allclose(result.reactions[current._fixed],
                               old_reactions[current._fixed],
                               rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("topology", ("Q4", "S3-V2D"))
def test_matching_one_shell_finite_elastic_history(topology):
    # The finite mixed owner uses the direct corotational Q4 response and the
    # elastic variational S3 response. Match that existing local law in the
    # bounded legacy owner; its optional variational Q4 law is distinct.
    legacy, current = _matching_owners(topology, direct_q4=topology == "Q4")
    actions = (NodalAction(
        "beam", 3, (0., .0001, .0002, 0., 0., 0.)),)
    for stop, factor in ((1, .5), (2, 1.)):
        old = legacy.solve(stop_after=stop)
        new = current.solve(actions, (factor,), max_iterations=20,
                            max_backtracks=12)
        assert old.status == "paused" and new.status == "completed"
        assert new.accepted_factors == (.5, 1.)[:stop]
        np.testing.assert_allclose(new.shell_displacement,
                                   old.state.shell_u, rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(new.beam_positions,
                                   old.state.mechanical.positions
                                   +old.state.mechanical.position_low,
                                   rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(new.beam_frames,
                                   old.state.mechanical.nodal_frames,
                                   rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(new.cell_frames,
                                   old.state.mechanical.cell_rotations,
                                   rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(new.cell_resultants,
                                   old.state.mechanical.resultants,
                                   rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(new.multipliers,
                                   old.state.multipliers,
                                   rtol=1e-9, atol=1e-11)
        _, old_residual, _, _ = legacy._evaluate(
            old.state, old.state.mechanical, old.state.shell_u,
            old.state.multipliers, stop)
        np.testing.assert_allclose(new.support_reactions[current._fixed],
                                   old_residual[current._fixed],
                                   rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("topology", ("Q4", "S3-V2D"))
def test_matching_one_shell_finite_tangent_through_actual_updates(topology):
    _, current = _matching_owners(topology)
    current.solve((NodalAction("beam", 3,
                               (0., .0001, .0002, 0., 0., 0.)),),
                  (.5, 1.), max_iterations=20, max_backtracks=12)
    driver = current._finite_driver
    state = driver._state
    residual, tangent, *_ = driver._assemble(
        state, 1., driver._shell_origins, driver._beam_origins,
        pattern=driver._pattern)
    count = current._K.shape[0]
    nodal_end = current._shell_count+6*len(current._beam_ids)
    rng = np.random.default_rng(20260929)
    direction = np.zeros(tangent.shape[0])
    direction[:nodal_end] = 1e-7*rng.normal(size=nodal_end)
    direction[nodal_end:count] = 1e-3*rng.normal(size=count-nodal_end)
    direction[count:] = 1e-3*rng.normal(size=len(direction)-count)
    direction[current._fixed] = 0.
    predicted = tangent@direction
    for amount in (.5, .25):
        plus = driver._advance(state, direction, amount)
        minus = driver._advance(state, direction, -amount)
        positive = driver._assemble(
            plus, 1., driver._shell_origins, driver._beam_origins,
            pattern=driver._pattern)[0]
        negative = driver._assemble(
            minus, 1., driver._shell_origins, driver._beam_origins,
            pattern=driver._pattern)[0]
        observed = (positive-negative)/(2*amount)
        for start, stop in ((0, current._shell_count),
                            (current._shell_count, count),
                            (count, len(direction))):
            error = np.linalg.norm((observed-predicted)[start:stop])
            scale = max(np.linalg.norm(predicted[start:stop]), 1e-10)
            assert error/scale <= 1e-6
    assert np.isfinite(residual).all()
