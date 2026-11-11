"""Development checks for the optional BL5 reference cell-average density.

These small work/tangent/restart checks do not establish uniform stability,
transfer convergence, or qualification of a multiplier-resolution policy.
"""
from dataclasses import replace

import numpy as np
import pytest

from anyfesolver import b3_ge, stiffened_shell as api
from anyfesolver._ge_beam3_retained_generalized_state import Mechanical
from anyfesolver._stiffened_shell_l1_assembly_experimental import _sample_rows
from test_stiffened_shell_bl5_affine_supports import candidate
from test_stiffened_shell_l1_assembly_experimental import _bl5_joint_mesh_definition


BASIS = "discontinuous_p1_cell_average"


def projected_definition(shell_edges=4, halfcells=4):
    base = candidate(shell_edges, halfcells)
    return replace(base, lines=(replace(base.lines[0], basis=BASIS),))


def test_supported_broken_translation_is_visible_to_fine_p1_but_not_p0():
    """A weak P0 null is not a zero-gap or zero-energy physical connection."""
    base = _bl5_joint_mesh_definition(4, 4, 1)
    p1 = replace(base, lines=(replace(base.lines[0], basis="continuous_p1"),))
    coarse = api.create_analysis(definition=base)
    fine = api.create_analysis(definition=p1)
    try:
        assert coarse._K.shape == fine._K.shape
        assert len(coarse._beam_ids) == 5
        amplitude = 1e-4
        q = np.zeros(coarse._K.shape[0])
        for i in range(len(coarse._beam_ids)):
            q[coarse._shell_count + 6*i + 1] = amplitude*(-1.)**i
        assert np.count_nonzero(q[coarse._fixed]) == 0

        positions = np.array([coarse._beam_nodes[n][0] for n in coarse._beam_ids])
        state = Mechanical(positions + q[coarse._shell_count:
            coarse._shell_count + 6*len(coarse._beam_ids)].reshape(-1, 6)[:, :3],
            np.zeros_like(positions), np.tile(np.eye(3), (len(positions), 1, 1)),
            np.tile(np.eye(3), (len(coarse._beam_parts), 2, 1, 1)),
            np.zeros((len(coarse._beam_parts), 18)))
        gaps = np.array([op.evaluate(np.zeros((2, 6)), state, np.zeros(5)).gap
                         for op, *_ in coarse._line_samples])
        assert np.max(np.abs(gaps[:, 1])) > amplitude/2
        integrated = np.zeros(coarse._C.shape[0])
        supports = len(base.lines[0].beam_halfcells)
        for gap, (_, _, weight, edge, _) in zip(gaps, coarse._line_samples):
            for channel in range(5):
                integrated[channel*supports + edge] += weight*gap[channel]
        np.testing.assert_allclose(integrated, coarse._C @ q, rtol=0., atol=1e-13)
        assert np.max(np.abs(coarse._C @ q)) < 1e-13
        assert np.max(np.abs(fine._C @ q)) > amplitude/100

        # Eliminate only stationary resultants for a physical strain-energy
        # check; they are not arbitrary positive-energy coordinates.
        physical = (coarse._shell_count + 6*len(coarse._beam_ids) +
                    6*len(coarse._beam_parts))
        k = coarse._K.toarray()
        kp = k[:physical, :physical] - k[:physical, physical:] @ np.linalg.solve(
            k[physical:, physical:], k[physical:, :physical])
        assert float(q[:physical] @ kp @ q[:physical]) > 0.
    finally:
        coarse.close()
        fine.close()


def test_distinct_identity_and_complete_member_admission():
    base = _bl5_joint_mesh_definition(4, 4, 1)
    old = replace(base.lines[0], basis="discontinuous_p1", multiplier_span=2)
    line = replace(old, basis=BASIS)
    assert line.multiplier_space == "BL5_P1_discontinuous_cell_average_span_2"
    assert line.multiplier_space != old.multiplier_space
    assert old.multiplier_space == "BL5_P1_discontinuous_span_2"
    assert line.descriptor()["basis"] == BASIS
    assert old.descriptor()["basis"] == "discontinuous_p1"
    assert "basis" not in base.lines[0].descriptor()
    assert base.lines[0].multiplier_space == "BL5_P0_span_1"
    for span in (1, 3):
        source = replace(line, beam_halfcells=tuple((i, j) for i in range(3)
                                                   for j in (0, 1)))
        with pytest.raises(ValueError, match="complete native members"):
            replace(source, multiplier_span=span)
    with pytest.raises(ValueError, match="complete native members"):
        replace(line, beam_halfcells=((0, 1), (1, 0), (1, 1), (2, 0)))
    with pytest.raises(ValueError, match="ordered even-edge"):
        replace(line, beam_halfcells=tuple((i, j) for i in range(3)
                                         for j in (0, 1)), multiplier_span=4)
    for coordinate in (None, np.nan, -.01, 1.01):
        with pytest.raises(ValueError, match="support coordinate"):
            _sample_rows(line, 0, .4, 0, .4, 0, coordinate)


@pytest.mark.parametrize("nonuniform", (False, True))
def test_physical_cell_means_all_channels_and_common_row_projection(nonuniform):
    definition = projected_definition(4, 8)
    xs = np.linspace(0., 1., 9)
    cell_means = np.tile((-.75, -.25, .25, .75), 2)
    if nonuniform:
        # True quadratic-member midpoints; member lengths vary within a support.
        xs = np.array((0., .05, .1, .25, .4, .5, .6, .8, 1.))
        cell_means = np.array((-.875, -.625, -.125, .625, -5/6, -.5, 0., 2/3))
        beams = []
        for i, old in enumerate(definition.beams):
            element, inertia = old.instantiate()
            points = np.column_stack((xs[2*i:2*i+3], np.zeros(3), np.full(3, .0675)))
            beams.append(b3_ge.define_beam("b3-ge", i+1,
                (101+2*i, 102+2*i, 103+2*i), points,
                np.tile(np.eye(3), (3, 1, 1)), element.section, inertia))
        definition = replace(definition, beams=tuple(beams))
    line = definition.lines[0]
    supports, span = 2, 4
    # Integrate the affine reference coordinate analytically on each cell.
    derived = []
    for edge, (left, right) in enumerate(zip(xs[:-1], xs[1:])):
        start = span*(edge//span)
        a, b = xs[start], xs[start+span]
        integral = ((right-a)**2-(left-a)**2)/(b-a)-(right-left)
        derived.append(integral/(right-left))
    np.testing.assert_allclose(derived, cell_means, rtol=0., atol=2e-15)
    common_definition = replace(definition, lines=(replace(line, basis="p0",
                                                          multiplier_span=1),))
    old_definition = replace(definition, lines=(replace(line, basis="discontinuous_p1"),))
    owner = api.create_analysis(definition=definition)
    common = api.create_analysis(definition=common_definition)
    old = api.create_analysis(definition=old_definition)
    try:
        assert owner.capabilities["status"] == "EXPERIMENTAL_NOT_QUALIFIED"
        assert owner._C.shape[0] == 10*supports == old._C.shape[0]
        projection = np.zeros((10*supports, 5*8))
        for channel in range(5):
            for edge, xi in enumerate(cell_means):
                row = 2*(channel*supports+edge//span)
                projection[row:row+2, channel*8+edge] = (1., xi)
        expected = projection@common._C.toarray()
        np.testing.assert_allclose(owner._C.toarray(), expected, rtol=0., atol=4e-14)
        assert np.linalg.norm(owner._C.toarray()-old._C.toarray()) > 1e-3
        integrals = np.zeros((supports, 2))
        measures = np.zeros(8)
        samples_per_cell = np.zeros(8, dtype=int)
        for sample, (rows, basis), common_sample in zip(
                owner._line_samples, owner._line_terms, common._line_samples):
            local, slots, weight, edge, alpha = sample
            np.testing.assert_array_equal(slots, common_sample[1])
            assert (weight, edge, alpha) == common_sample[2:]
            np.testing.assert_array_equal(local.reference_arm, common_sample[0].reference_arm)
            xi, support = cell_means[edge], edge//span
            interpolation = np.zeros((5, 10))
            expected_rows = []
            for channel in range(5):
                interpolation[channel, 2*channel:2*channel+2] = (1., xi)
                expected_rows.extend((2*(channel*supports+support),
                                      2*(channel*supports+support)+1))
            np.testing.assert_array_equal(rows, expected_rows)
            np.testing.assert_allclose(basis, interpolation, rtol=0., atol=2e-15)
            integrals[support] += weight*np.array((1., xi))
            measures[edge] += weight
            samples_per_cell[edge] += 1
            # The actual point remains within one cell and one shell edge.
            x = xs[edge]+alpha*(xs[edge+1]-xs[edge])
            assert xs[edge] < x < xs[edge+1]
        np.testing.assert_allclose(measures, np.diff(xs), rtol=0., atol=2e-15)
        np.testing.assert_allclose(integrals,
            ((xs[4]-xs[0], 0.), (xs[8]-xs[4], 0.)), rtol=0., atol=2e-15)
        assert len(owner._line_samples) == len(common._line_samples)
        assert samples_per_cell.sum() == 2*(len(np.unique(np.r_[xs, np.linspace(0., 1., 5)]))-1)
        if nonuniform:
            assert samples_per_cell.max() == 4
        rng = np.random.default_rng(734)
        q = rng.normal(size=owner._C.shape[1])
        multiplier = rng.normal(size=owner._C.shape[0])
        np.testing.assert_allclose(multiplier@owner._C@q,
            (projection.T@multiplier)@common._C@q, rtol=2e-14, atol=1e-14)
    finally:
        owner.close(); common.close(); old.close()


def test_projected_multiplier_curvature_through_actual_state_update():
    from anyfesolver._stiffened_shell_ge_finite_experimental import FinitePointDriver
    definition = replace(projected_definition(),
                         q4_pullback_policy=api.Q4_ALGORITHMIC_PULLBACK_V1)
    owner = api.create_analysis(definition=definition)
    try:
        driver = FinitePointDriver(owner)
        count = owner._K.shape[0]
        rng = np.random.default_rng(192)
        delta = rng.normal(size=count+owner._C.shape[0])*.002
        delta[count:] = rng.normal(size=owner._C.shape[0])*50000.
        state = driver._advance(driver._state, delta, 1.)
        zero = replace(state, multipliers=np.zeros_like(state.multipliers))
        pattern = np.zeros(count)
        def evaluate(s):
            return driver._assemble(s, 0., driver._shell_origins,
                                    driver._beam_origins, pattern=pattern)[:2]
        r, k = evaluate(state)
        r0, k0 = evaluate(zero)
        direction = rng.normal(size=len(r)); direction[count:] = 0.
        expected = (k-k0)@direction
        assert np.linalg.norm(expected) > 1.
        errors = []
        for step in (1e-5, 3e-6, 1e-6):
            residuals = []
            for sign in (-1., 1.):
                moved = driver._advance(state, direction, sign*step)
                moved0 = replace(moved, multipliers=np.zeros_like(moved.multipliers))
                residuals.append(evaluate(moved)[0]-evaluate(moved0)[0])
            numerical = (residuals[1]-residuals[0])/(2.*step)
            errors.append(np.linalg.norm(numerical-expected)/np.linalg.norm(expected))
        assert min(errors) < 1e-6, errors
    finally:
        owner.close()


def test_finite_density_checkpoint_identity_and_same_candidate_restart():
    definition = replace(projected_definition(),
                         q4_pullback_policy=api.Q4_ALGORITHMIC_PULLBACK_V1)
    old_definition = replace(definition,
        lines=(replace(definition.lines[0], basis="discontinuous_p1"),))
    owner = api.create_analysis(definition=definition)
    restored = api.create_analysis(definition=definition)
    old = api.create_analysis(definition=old_definition)
    action = (api.NodalAction("beam", 105, (0., 0., 1., 0., 0., 0.)),)
    try:
        result = owner.solve(action, targets=(.5, 1.))
        assert result.status == "completed" and result.accepted_factors == (.5, 1.)
        assert result.free_equilibrium_ratio < 1e-8 and result.constraint_max < 1e-9
        assert owner._C.shape == old._C.shape
        for sample, record in zip(owner._line_samples, result.beam_shell_line_work):
            edge = sample[3]
            support, xi = edge//2, (-.5, .5)[edge % 2]
            expected = np.array([result.multipliers[2*(channel*2+support)]+
                xi*result.multipliers[2*(channel*2+support)+1] for channel in range(5)])
            np.testing.assert_allclose(record.multiplier_density, expected, rtol=1e-14, atol=1e-13)
        saved = owner.checkpoint()
        with pytest.raises(ValueError, match="incompatible experimental.*checkpoint"):
            old.restore(saved)
        old_result = old.solve(action, targets=(.5,))
        assert old_result.status == "completed"
        with pytest.raises(ValueError, match="incompatible experimental.*checkpoint"):
            restored.restore(old.checkpoint())
        repeated = restored.restore(saved)
        assert restored.checkpoint() == saved
        np.testing.assert_array_equal(repeated.beam_positions, result.beam_positions)
        for a, b in zip(repeated.beam_shell_line_work, result.beam_shell_line_work):
            np.testing.assert_array_equal(a.multiplier_density, b.multiplier_density)
        direct = owner.solve(action, targets=(.75,))
        continued = restored.solve(action, targets=(.75,))
        assert direct.status == continued.status == "completed"
        np.testing.assert_array_equal(continued.beam_positions, direct.beam_positions)
        assert restored.checkpoint() == owner.checkpoint()
    finally:
        owner.close(); restored.close(); old.close()
