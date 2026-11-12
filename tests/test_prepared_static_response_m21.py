"""M21: batched staged unit probes for the vectorized affine maps."""
import numpy as np
from anyfesolver import RecoveryConfig, prepare_static_response_evaluator
from anyfesolver.prepared_static_recovery import owned_recovery_scope, _recover_owned_local
from anyfesolver import prepared_static_vectorized as vector
import prepared_static_response_m16_assessment  # noqa: F401  (adds scripts/ to sys.path)
from benchmark_static_batches import mixed_free_model, free_cases

ENVELOPE = RecoveryConfig(include_displacements=False, include_stresses=True, include_reactions=True)


def test_batched_probes_match_unit_probes_and_are_used_for_staged_q4():
    model, groups, balancing = mixed_free_model()
    patterns = {name: load for name, _time, load in free_cases(model, groups, balancing, [5, 99])}
    with prepare_static_response_evaluator(model, patterns, envelope_recovery_config=ENVELOPE,
                                           stress_execution='certified_vectorized') as owner:
        union = owner._library_elements(owner._snapshot)
        staged = {eid: item for eid, item in union.items() if item[1].staged_kernel is not None}
        assert staged, 'fixture must contain staged Q4 elements'
        with owned_recovery_scope(tuple(recipe for _mapping, recipe in union.values())):
            for eid, (mapping, recipe) in union.items():
                size = int(mapping.size)
                zero = _recover_owned_local(recipe, np.zeros(size))
                _order, linear, constants, _derived, count = vector._schema(eid, zero)
                base = vector._vector(zero, linear, count)
                batched = vector._batched_unit_probes(recipe, size, linear, count, base, constants)
                if eid not in staged:
                    assert batched is None
                    continue
                unit = np.empty((count, size))
                for k in range(size):
                    probe = np.zeros(size)
                    probe[k] = 1.0
                    unit[:, k] = vector._vector(_recover_owned_local(recipe, probe), linear, count) - base
                scale = max(float(np.max(np.abs(unit))), 1e-300)
                assert np.max(np.abs(batched - unit)) <= 1e-13 * scale
        info = owner.diagnostics()['source_response']
        expected = sum(1 + 3 if eid in staged else int(m.size) + 3 for eid, (m, _r) in union.items())
        assert info['vectorized_probes'] == expected
        assert info['vectorized_worst_verification_residual'] <= 1e-9
