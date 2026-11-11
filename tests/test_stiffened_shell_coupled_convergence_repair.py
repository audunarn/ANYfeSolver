"""Structural characterization of the coupled p0 mixed tangent near-null mode.

The coupled p0 adaptive exhaustion is driven by an oversized exact Newton
step whose work content localizes in the line multiplier (force density)
coordinates. These tests pin, through the production path, the structural
facts any repair of that mechanism must respect:

- the BL5 line work is exactly invariant under shell drilling: the line
  operator's response depends on the shell frames only through the director
  column, so lines can contribute no drilling stiffness at any order, while
  a director tilt does change the response (control);
- the assembled mixed tangent has an exactly zero multiplier-by-multiplier
  block (the gap equations do not depend on the multipliers), while the
  multiplier residual rows are live at a perturbed state;
- at the captured failed parent (skipped without the local fixture): the
  Newton direction's work content localizes in the multiplier group, and
  the constraint force/gap blocks and the displacement block all keep full
  rank - the near-null mode is not a constraint-basis or displacement
  rank defect.

No singular-value quotient is asserted anywhere; the rank checks are
structural facts and the localization assertion carries nine orders of
measured margin (0.999999... vs 0.99).
"""
import json
from pathlib import Path

import numpy as np
import pytest

from anyfesolver import stiffened_shell as ss
from anyfesolver._ge_beam3_p5.algebra import rotation
from anyfesolver._ge_beam3_retained_generalized_state import Mechanical
from anyfesolver._stiffened_shell_ge_finite_experimental import (
    FinitePointDriver, _State, _owned)
from anyfesolver._stiffened_shell_rotation_charts import physical_frames
from test_stiffened_shell_bl5_coupled_material import coupled
from test_stiffened_shell_bl5_finite import _action

FIXTURE = (Path(__file__).resolve().parents[1]/'.restart-ci'/
           'windows-completion-14/failed-states/coupled-p0')


def _owner():
    return ss.create_analysis(definition=coupled('p0', simultaneous_yield_fixture=True))


def test_bl5_line_work_is_exactly_invariant_under_shell_drilling(request):
    """Drilling rotates the frame's in-plane axes; the line work cannot see it."""
    owner = _owner()
    request.addfinalizer(owner.close)
    driver = FinitePointDriver(owner)
    state = driver._state
    shell_frames = physical_frames(state.q[:owner._shell_count], state.shell_anchors)
    beam_positions, beam_low = driver._beam_position_pair(state)
    mechanical = Mechanical(beam_positions, beam_low, state.frames,
                            state.cells, state.resultants)
    density = np.array((137.5, -80.25, 3.25, -.5, 250.))
    for sample, (_line_rows, basis) in zip(owner._line_samples, owner._line_terms):
        operator, slots, _weight, _edge, _alpha = sample
        shell_nodes = slots[:12:6]//6
        frames = shell_frames[shell_nodes]
        total = state.q[slots[:12]].reshape(2, 6)
        base = operator.evaluate_spatial(total, frames, mechanical, density)
        drilled = frames.copy()
        for i in range(2):
            axis = frames[i][:, 2]
            turned = rotation((.37 if i else -.61)*axis)@frames[i]
            drilled[i, :, :2] = turned[:, :2]
        same = operator.evaluate_spatial(total, drilled, mechanical, density)
        np.testing.assert_array_equal(base.gap, same.gap)
        np.testing.assert_array_equal(base.gradient, same.gradient)
        np.testing.assert_array_equal(base.hessian, same.hessian)
        np.testing.assert_array_equal(base.gap_jacobian, same.gap_jacobian)
        tilted = frames.copy()
        for i in range(2):
            axis = frames[i][:, 0]
            turned = rotation((.29 if i else -.43)*axis)@frames[i]
            tilted[i, :, :2] = turned[:, :2]
            tilted[i, :, 2] = turned[:, 2]
        moved = operator.evaluate_spatial(total, tilted, mechanical, density)
        assert not np.array_equal(base.gap, moved.gap)


def test_assembled_mixed_tangent_multiplier_block_is_exactly_zero(request):
    """The gap rows never depend on the multipliers, at a live perturbed state."""
    owner = _owner()
    request.addfinalizer(owner.close)
    driver = FinitePointDriver(owner)
    count = owner._K.shape[0]
    rng = np.random.default_rng(819)
    delta = rng.normal(size=count+owner._C.shape[0])*.002
    delta[count:] = rng.normal(size=owner._C.shape[0])*50000.
    state = driver._advance(driver._state, delta, 1.)
    residual, matrix, *_ = driver._assemble(
        state, 0., driver._shell_origins, driver._beam_origins,
        pattern=np.zeros(count))
    block = np.asarray(matrix[count:, count:].todense(), dtype=float)
    assert not np.any(block)
    assert np.any(residual[count:] != 0.)


@pytest.mark.skipif(not (FIXTURE/'metadata.json').exists(),
                    reason='captured coupled-p0 parent fixture is local only')
def test_saved_parent_direction_localizes_in_multiplier_work(request):
    """The oversized exact Newton step is multiplier work; all blocks keep rank."""
    meta = json.loads((FIXTURE/'metadata.json').read_text())
    owner = _owner()
    request.addfinalizer(owner.close)
    owner.restore((FIXTURE/'accepted-prefix.bin').read_bytes())
    driver = owner._finite_driver
    before = owner.checkpoint()
    with np.load(FIXTURE/'failed-parent.npz', allow_pickle=False) as saved:
        values = {name: _owned(saved[name]) for name in saved.files}
    parent = _State(**values, **{name: None for name in meta['absent_fields']})
    staged = ss.StagedDeadLoadProgramme(owner.preflight()['identity'],
                                        (_action(700.), _action(-500.)))
    pattern, programme = driver._prepare_load(staged, (2.,))
    residual, matrix, relative, _gap, *_ = driver._assemble(
        parent, meta['factor'], driver._shell_origins, driver._beam_origins,
        pattern=pattern, applied_load=programme.load_at(meta['factor']),
        load_programme=programme)
    assert abs(relative-meta['baseline_relative']) <= 1e-9*meta['baseline_relative']
    count = owner._K.shape[0]
    free = np.r_[owner._free, np.arange(count, count+len(parent.multipliers))]
    weights = np.r_[owner._coordinate_scale, 1./owner._constraint_scale]
    wf = weights[free]
    assert not np.any(np.asarray(matrix[count:, count:].todense(), dtype=float))
    dense = np.asarray(matrix[free, :][:, free].todense(), dtype=float)
    work = dense*wf[None, :]*wf[:, None]
    direction = np.linalg.solve(work, (-residual[free])*wf)
    multipliers = np.array([index >= count for index in free])
    total = float(np.linalg.norm(direction))
    assert total > 0.
    assert float(np.linalg.norm(direction[multipliers]))/total > .99
    displacement = ~multipliers
    assert np.linalg.matrix_rank(work[np.ix_(displacement, displacement)]) == int(displacement.sum())
    assert np.linalg.matrix_rank(work[np.ix_(displacement, multipliers)]) == int(multipliers.sum())
    assert np.linalg.matrix_rank(work[np.ix_(multipliers, displacement)]) == int(multipliers.sum())
    assert owner.checkpoint() == before
