"""Fixed-predecessor actual-update derivative, not aggregate qualification."""
import json
from math import fsum

import numpy as np
import pytest

from anyfesolver import b3_ge, stiffened_shell as ss
from anyfesolver._stiffened_shell_beam_history import arrays as native_history_arrays
from test_stiffened_shell_distributing_native_material import definition, packed


@pytest.mark.parametrize("mixed", (False, True), ids=("fibre-j2", "mixed-native-j2"))
def test_yielded_actual_update_global_tangent(mixed, record_property, monkeypatch):
    model = definition(coupled=True, mixed=mixed, yielding="axial")
    owner = ss.create_analysis(definition=model)
    resumed = ss.create_analysis(definition=model)
    try:
        result = owner.solve(model.cyclic_reference_actions, (.25, .5, .75, 1.),
            max_iterations=24, max_backtracks=16)
        before = owner.checkpoint()
        kind, generation, arrays = packed(owner)
        fibre_increment = np.array([fsum((*a[2:], *-b[2:])) for a, b in
            zip(arrays["accepted_fibres"], arrays["origin_fibres"])])
        assert np.max(fibre_increment) > 0.
        assert np.max(arrays["shell_alpha"]-arrays["shell_origin_alpha"]) > 0.
        if mixed:
            assert np.max(arrays["accepted_accumulated"]-arrays["origin_accumulated"]) > 0.
        driver = owner._finite_driver
        def evaluate(state):
            return driver._assemble(state, 1., driver._shell_predecessors,
                driver._beam_predecessors, pattern=driver._pattern)
        residual, tangent, *_ = evaluate(driver._state)
        again = evaluate(driver._state)
        np.testing.assert_array_equal(residual, again[0])
        np.testing.assert_array_equal(tangent.toarray(), again[1].toarray())
        resumed.restore(before)
        other = resumed._finite_driver
        restored = other._assemble(other._state, 1., other._shell_predecessors,
            other._beam_predecessors, pattern=other._pattern)
        np.testing.assert_array_equal(residual, restored[0])
        np.testing.assert_array_equal(tangent.toarray(), restored[1].toarray())

        def active_branches(response):
            beam = dict(native_history_arrays(owner._beam_parts, response[5], "trial"))
            fibres = np.array([fsum((*a[2:], *-b[2:])) > 0. for a, b in
                zip(beam["trial_fibres"], arrays["origin_fibres"])])
            generalized = np.array([fsum((*a, *-b)) > 0. for a, b in
                zip(beam["trial_accumulated"], arrays["origin_accumulated"])])
            shell = np.concatenate([candidate["alpha"]-origin["alpha"] > 0.
                for candidate, origin in zip(response[4], driver._shell_predecessors)])
            return fibres, generalized, shell
        branches = active_branches(again)

        count = owner._K.shape[0]
        beam_end = owner._shell_count+6*len(owner._beam_ids)
        rng = np.random.default_rng(818)
        direction = np.zeros(len(residual))
        direction[:beam_end] = rng.normal(size=beam_end)*1e-4
        for index in range(len(owner._beam_parts)):
            base = beam_end+24*index
            direction[base:base+6] = rng.normal(size=6)*1e-4
            resultants = direction[base+6:base+24].reshape(3, 6)
            resultants[:, :3] = rng.normal(size=(3, 3))*1000.
            resultants[:, 3:] = rng.normal(size=(3, 3))*30.
        multipliers = direction[count:].reshape(-1, 6)
        multipliers[:, :3] = rng.normal(size=multipliers[:, :3].shape)*1000.
        multipliers[:, 3:] = rng.normal(size=multipliers[:, 3:].shape)*30.
        direction[owner._fixed] = 0.
        weights = np.r_[owner._coordinate_scale, np.ones(owner._C.shape[0])]
        predicted = (tangent@direction)*weights
        groups = {
            "shell": owner._free[owner._free < owner._shell_count],
            "beam": owner._free[(owner._free >= owner._shell_count)&(owner._free < beam_end)],
            "retained": np.arange(beam_end, count),
            "constraints": np.arange(count, len(residual))}
        predicted_norms = {name:float(np.linalg.norm(predicted[rows])) for name, rows in groups.items()}
        assert all(value > 0. for value in predicted_norms.values())
        steps = (1e-4, 1e-5, 1e-6)
        errors = {name:[] for name in groups}
        for h in steps:
            plus = evaluate(driver._advance(driver._state, direction, h))
            minus = evaluate(driver._advance(driver._state, direction, -h))
            for response in (plus, minus):
                for expected, actual in zip(branches, active_branches(response)):
                    np.testing.assert_array_equal(actual, expected)
            difference = ((plus[0]-minus[0])/(2*h))*weights-predicted
            for name, rows in groups.items():
                errors[name].append(float(np.linalg.norm(difference[rows])/predicted_norms[name]))
        record_property("native_tangent", json.dumps(dict(kind=kind, generation=generation,
            fibre_increment_max=float(np.max(fibre_increment)), steps=steps,
            active_branch_counts=[int(np.count_nonzero(b)) for b in branches],
            branch_masks_unchanged=True,
            weighted_prediction_norms=predicted_norms, relative_errors=errors,
            physical_rows=beam_end, retained_rows=count-beam_end,
            multiplier_rows=len(residual)-count), sort_keys=True))
        # At least two adjacent step sizes must establish the finite-difference
        # plateau. Separate row groups prevent a large force block hiding gaps
        # or native retained-variable derivatives; no reference floor is added.
        assert any(all(errors[name][i] <= 1e-5 and errors[name][i+1] <= 1e-5
                       for name in groups) for i in range(len(steps)-1)), errors
        assert owner.checkpoint() == resumed.checkpoint() == before
        assert owner._generation == generation and owner.recover() is result
        def forbidden(*args, **kwargs):
            raise AssertionError("accepted native recovery repeated trial evaluation")
        with monkeypatch.context() as patch:
            patch.setattr(driver, "_assemble", forbidden)
            for element, _, _ in owner._beam_parts:
                patch.setattr(type(element.operator), "evaluate", forbidden)
                if type(element.section) is b3_ge.PhysicalFibreSection:
                    patch.setattr(type(element.operator), "recover_material", forbidden)
            assert owner.recover() is result
    finally:
        owner.close(); resumed.close()
