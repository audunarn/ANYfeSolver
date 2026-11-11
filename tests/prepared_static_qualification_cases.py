"""Preserved canonical-principal qualification counterexample; run explicitly.

This is separate from the isolated nonlinear-operator tests. It intentionally
retains the unmet canonical recovery comparison at its original tolerance.
Evidence: solver-finish-12, with 3/21 mismatches up to 5.39444e-11.
"""
import numpy as np
from test_prepared_static_evaluator import _shell_model, _shell_case
from anyfesolver import RecoveryConfig, prepare_static_evaluator
from anyfesolver.recovery import recover_stress_result
from anyfesolver.prepared_static_measures import shell_surface_tensor


def test_canonical_principal_recovery_qualification():
    model = _shell_model()
    case = _shell_case(0, pressure=2.)
    recovery = RecoveryConfig(components=['principal_stresses_top', 'principal_stresses_bottom'], include_stresses=True)
    with prepare_static_evaluator(model, recovery_config=recovery) as owner:
        block = next(owner.iter_batches([case]))
        raw = recover_stress_result(model, block.displacements[:,0], RecoveryConfig(include_stresses=True))
        for eid, fields in raw.element_stresses.items():
            for surface in ('top', 'bottom'):
                expected = np.linalg.eigvalsh(shell_surface_tensor(fields, surface))[..., ::-1]
                np.testing.assert_allclose(block.case_results[0]['element_stresses'][eid]['principal_stresses_'+surface], expected, rtol=1e-12, atol=1e-12)
