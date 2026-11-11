"""Independent moving-support oracle for the constrained pressure response."""

from types import SimpleNamespace

import numpy as np
from scipy import sparse

from anyfesolver.general_contact import linearized_pivot


class _MovingSupportGeometry:
    def evaluate(self, source, _target):
        coordinate = float(source[0, 0])
        area = np.array([1.0 + coordinate, 1.0])
        gap = np.array([-0.1 + 0.2*coordinate, -0.2 + 0.3*coordinate])
        return SimpleNamespace(
            reference_support=area,
            reference_weighted_gap=area*gap,
            normalized_gap=gap,
            overlap_signature=(1, 2),
        )


def test_schur_uses_fixed_multiplier_force_work_with_moving_support(monkeypatch):
    geometry = _MovingSupportGeometry()
    manager = SimpleNamespace(
        _positions=lambda displacement: (np.array([[displacement[0], 0.0, 0.0]]),
                                         np.zeros((1, 3))),
        normal_multipliers=np.ones(2),
        law=SimpleNamespace(
            geometry=geometry,
            evaluate=lambda source, target, pressure: SimpleNamespace(
                pressure=pressure-geometry.evaluate(source, target).normalized_gap,
                normalized_gap=geometry.evaluate(source, target).normalized_gap,
            ),
        ),
        map=sparse.csr_matrix(([1.0], ([0], [0])), shape=(6, 1)),
        stiffness=1.0,
        settings=SimpleNamespace(maximum_general_s3_energy_evaluations=100),
    )
    seen = {}

    def capture(_pressure, _gap, _active, response, **_kwargs):
        seen["schur"] = response
        return None

    monkeypatch.setattr(linearized_pivot, "propose_active_set_pivot", capture)
    linearized_pivot.refined_patch_pivot(
        manager, np.zeros(1), sparse.csr_matrix([[2.0]]),
        sparse.eye(1, format="csr"), np.zeros((0, 1)),
        difference_step=1.0e-6, maximum_pressure=10.0,
        pressure_tolerance=1.0e-3,
    )
    # At x=0, S=(1,1), g=(-.1,-.2), dW/dx=(.1,.3), dS/dx=(1,0).
    # The actual contact virtual work uses G=dW-g*dS=(.2,.3).
    # K=2 gives dq/dlambda=G/2, hence H=dg/dlambda=G G^T/2.
    independent = np.outer([0.2, 0.3], [0.2, 0.3])/2.0
    np.testing.assert_allclose(seen["schur"], independent, rtol=1.0e-9,
                               atol=1.0e-11)


def test_schur_matches_independently_reequilibrated_multiplier_directions(monkeypatch):
    geometry = _MovingSupportGeometry()
    manager = SimpleNamespace(
        _positions=lambda displacement: (np.array([[displacement[0], 0.0, 0.0]]),
                                          np.empty((0, 3))),
        normal_multipliers=np.ones(2),
        law=SimpleNamespace(
            geometry=geometry,
            evaluate=lambda source, target, pressure: SimpleNamespace(
                pressure=pressure, normalized_gap=geometry.evaluate(
                    source, target,
                ).normalized_gap,
            ),
        ),
        map=sparse.csr_matrix([[1.0], [0.0], [0.0]]),
        settings=SimpleNamespace(maximum_general_s3_energy_evaluations=100),
        stiffness=1.0,
    )
    seen = {}

    def capture(_pressure, _gap, _active, response, **_kwargs):
        seen["schur"] = response
        return None

    monkeypatch.setattr(linearized_pivot, "propose_active_set_pivot", capture)
    linearized_pivot.refined_patch_pivot(
        manager, np.zeros(1), sparse.csr_matrix([[1.8]]),
        sparse.eye(1, format="csr"), np.zeros((0, 1)),
        difference_step=1.0e-6, maximum_pressure=10.0,
        pressure_tolerance=1.0e-3,
    )

    def independently_equilibrated_gap(pressure):
        # Solve 2q - sum(lambda_i S_i(q) dg_i/dq) - F = 0 with F=-0.5.
        q = (0.2*pressure[0]+0.3*pressure[1]-0.5)/(2.0-0.2*pressure[0])
        return np.array([-0.1+0.2*q, -0.2+0.3*q])

    base = np.ones(2)
    step = 1.0e-5
    independent = np.column_stack([
        (independently_equilibrated_gap(base+step*axis)
         - independently_equilibrated_gap(base-step*axis))/(2*step)
        for axis in np.eye(2)
    ])
    np.testing.assert_allclose(seen["schur"], independent, rtol=1.0e-9,
                               atol=1.0e-11)


def test_supported_schur_uses_five_coupled_pressure_modes():
    class Geometry:
        def evaluate(self, source, _target):
            x = source[:, 0]
            area = 1.0+0.1*x
            gap = -0.1+0.2*x
            return SimpleNamespace(
                reference_support=area,
                reference_weighted_gap=area*gap,
                normalized_gap=gap,
                overlap_signature=(1,),
            )

    geometry = Geometry()
    manager = SimpleNamespace(
        _positions=lambda displacement: (
            np.column_stack((displacement, np.zeros((5, 2)))),
            np.empty((0, 3)),
        ),
        normal_multipliers=np.ones(5),
        law=SimpleNamespace(
            geometry=geometry,
            evaluate=lambda source, target, pressure: SimpleNamespace(
                reference_support=geometry.evaluate(source, target).reference_support,
                normalized_gap=geometry.evaluate(source, target).normalized_gap,
                pressure=np.maximum(0.0, pressure-geometry.evaluate(
                    source, target,
                ).normalized_gap),
            ),
        ),
        map=sparse.csr_matrix(np.eye(15)[:, [0, 3, 6, 9, 12]]),
        settings=SimpleNamespace(maximum_general_s3_energy_evaluations=100),
        stiffness=1.0,
    )
    proposal = linearized_pivot.refined_patch_supported_schur(
        manager, np.zeros(5), sparse.diags([2.0]*5, format="csr"),
        sparse.eye(5, format="csr"), np.zeros((0, 5)),
        difference_step=1.0e-6, maximum_pressure=20.0,
        maximum_pressure_step=10.0, maximum_modes=5,
    )
    assert proposal is not None
    np.testing.assert_allclose(proposal.multipliers, 6.0,
                               rtol=1.0e-9, atol=1.0e-9)
    assert proposal.active_indices == (0, 1, 2, 3, 4)
    # Re-equilibrate the independent nonlinear spring/contact equations.
    # At lambda=1, q=0 is balanced by a fixed external force of -0.2 per DOF.
    q = (0.2*proposal.multipliers-0.2)/(2.0-0.02*proposal.multipliers)
    actual_gap = -0.1+0.2*q
    actual_residual = np.maximum(0.0, proposal.multipliers-actual_gap)
    actual_residual -= proposal.multipliers
    assert np.max(np.abs(actual_residual)) < 0.1*proposal.predicted_residual_before


def test_pressure_response_rejects_a_clipping_edge_tie():
    geometry = SimpleNamespace(
        source_face_indices=(np.array((0, 1, 2, 3)),),
        target_face_indices=(np.array((0, 1, 2)),),
    )
    source = np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                       (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)))
    target = np.array(((0.2, 0.2, 0.1), (0.7, 0.2, 0.1),
                       (0.2, 0.7, 0.1)))
    assert linearized_pivot._projected_clip_signature(
        geometry, source, target,
    ) is not None
    target[0, 0] = 0.0
    assert linearized_pivot._projected_clip_signature(
        geometry, source, target,
    ) is None


def test_distant_edge_line_extension_does_not_block_pressure_response():
    geometry = SimpleNamespace(
        source_face_indices=(np.array((0, 1, 2, 3)),),
        target_face_indices=(np.array((0, 1, 2)),),
    )
    source = np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                       (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)))
    target = np.array(((2.0, 0.0, 0.1), (3.0, 0.0, 0.1),
                       (2.5, 1.0, 0.1)))
    assert linearized_pivot._projected_clip_signature(
        geometry, source, target,
    ) is not None
