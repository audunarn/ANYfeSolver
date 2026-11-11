"""Restricted coupled native B3-GE and deformable S3 static development."""

from hashlib import sha256
import json

import numpy as np
import pytest

from anyfesolver import create_shell_element
from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition
from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_retained_nodal_loading import NodalDeadForces, Program
from anyfesolver._ge_beam3_p5_seeded.core import canonical, sha
from anyfesolver.boundary import BoundaryCondition, FixedSupport
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.native_retained_line_s3 import (
    NativeRetainedRoundLineS3Interior,
)
from anyfesolver.general_contact.native_retained_line_s3_work import (
    NativeRetainedLineS3Work,
)
from anyfesolver.general_contact.native_retained_s3_static import (
    NativeRetainedRoundS3Analysis,
)
from anyfesolver.general_contact.overlap import reference_shell_face
from anyfesolver.general_contact.s3_skin_work import S3SkinWorkAdapter
from test_ge_beam3_native_generalized_modal import make


def _analysis(*, force=-0.01, iterations=24, shell_z=-0.051,
              shell_elastic_modulus=1e5, shell_yield_stress=0.0,
              curved=False, shell_supports=True, targets=(1.0,)):
    native_model, _, inertias = make(curved, clamped=True)
    native_element = native_model.mesh.elements[1]
    definition = NativeBeamDefinition.capture(native_element, inertias[1])
    program = Program(
        targets, DistributedPattern(LinePattern(()), ()),
        NodalDeadForces(((3, 0.0, 0.0, float(force)),)),
        max_iterations=iterations,
    )
    shell = FEModel("native_s3_static")
    shell.add_material("elastic", shell_elastic_modulus, 0.3, density=1.0,
                       yield_stress=shell_yield_stress)
    for node, (x, y) in zip((4, 5, 6),
                            ((-5.0, -5.0), (5.0, -5.0), (0.0, 10.0))):
        shell.add_node(node, x, y, shell_z-0.025)
    shell.add_element(21, create_shell_element(
        21, (4, 5, 6), "elastic", thickness=0.05,
        formulation="e4-pl-s3-v2d", reference_normal=(0, 0, 1),
        reference_surface_offset=0.0,
    ))
    if shell_supports:
        shell.add_boundary_condition(FixedSupport("edge", [4, 6]))
        shell.add_boundary_condition(BoundaryCondition(
            "moving_vertex", [5],
            {dof: 0.0 for dof in ("ux", "uy", "rx", "ry", "rz")},
        ))
    skin = S3SkinWorkAdapter.from_model(shell, 7, 21, "top")
    kernel = NativeRetainedRoundLineS3Interior(
        1, native_element.operator.reference, (1, 2, 3),
        21, reference_shell_face(shell, 7, 21, "top"), (4, 5, 6),
        radius_m=0.05, line_stiffness_n_per_m2=1000.0,
        rim_clearance_m=0.1, geometry_tolerance_m=1e-10,
        energy_tolerance_j=1e-10, force_tolerance_n=1e-9,
        tangent_tolerance_n_per_m=1e-7,
        maximum_integration_intervals=100,
    )
    return NativeRetainedRoundS3Analysis(
        (definition,), tuple(native_model.boundary_conditions), program,
        shell, NativeRetainedLineS3Work(kernel, skin),
        gap_tolerance_m=0.001, equilibrium_tolerance=1e-8,
        maximum_swept_intervals=128,
    )


def test_coupled_native_s3_static_contact_and_wrench():
    owner = _analysis()
    result = owner.solve()
    assert result.status == "completed", result.failure
    accepted = result.accepted
    assert accepted.completed_targets == 1
    assert accepted.contact.contact.beam_force_n[2] > 0
    assert np.linalg.norm(accepted.shell_displacement) > 1e-10
    np.testing.assert_allclose(
        accepted.contact.contact.beam_force_n+accepted.contact.contact.shell_force_n,
        0, atol=1e-8,
    )
    assert accepted.contact.contact.minimum_gap_m >= -owner.gap_tolerance
    _, _, metrics, _, _ = owner._assemble(
        accepted.mechanical, accepted.shell_displacement,
        1.0, accepted.native_histories,
    )
    assert max(metrics) < owner.equilibrium_tolerance


def test_initial_interference_and_failed_solve_preserve_committed_state():
    with pytest.raises(ValueError, match="initial native/S3 contact penetration"):
        _analysis(shell_z=-0.04)
    owner = _analysis(iterations=1)
    result = owner.solve()
    assert result.status == "failed"
    assert result.accepted.completed_targets == 0
    np.testing.assert_array_equal(result.accepted.shell_displacement,
                                  owner.initial.shell_displacement)
    np.testing.assert_array_equal(result.accepted.mechanical.positions,
                                  owner.initial.mechanical.positions)
    assert sha(result.accepted.native_histories) == sha(owner.initial.native_histories)


def test_assembled_beam_shell_cross_tangent():
    owner = _analysis()
    result = owner.solve()
    assert result.status == "completed", result.failure
    accepted = result.accepted
    residual, tangent, _, _, _ = owner._assemble(
        accepted.mechanical, accepted.shell_displacement,
        1.0, accepted.native_histories,
    )
    n = owner.context.count
    assert np.max(np.abs(tangent[:n, n:])) > 1e-6
    assert np.max(np.abs(tangent[n:, :n])) > 1e-6
    direction = np.zeros(len(residual))
    direction[8] = -0.012
    direction[14] = 0.017
    direction[n+owner.shell_free[0]] = -0.009
    step = 1e-5
    positive = owner.context.advance(accepted.mechanical,
                                     step*direction[:n])
    negative = owner.context.advance(accepted.mechanical,
                                     -step*direction[:n])
    plus = owner._assemble(
        positive, accepted.shell_displacement+step*direction[n:],
        1.0, accepted.native_histories,
    )[0]
    minus = owner._assemble(
        negative, accepted.shell_displacement-step*direction[n:],
        1.0, accepted.native_histories,
    )[0]
    observed = (plus-minus)/(2*step)
    predicted = tangent@direction
    assert np.linalg.norm(observed-predicted)/max(1.0, np.linalg.norm(predicted)) < 1e-5


def test_curved_native_internal_contact_work_survives_coupled_solve():
    owner = _analysis(curved=True)
    result = owner.solve()
    assert result.status == "completed", result.failure
    assert np.linalg.norm(result.accepted.contact.contact.residual[18:24]) > 1e-7
    assert np.linalg.norm(result.accepted.shell_displacement) > 1e-10


def test_unsupported_shell_material_mechanism_and_mutation_fail():
    with pytest.raises(ValueError, match="isotropic elastic"):
        _analysis(shell_yield_stress=1.0)
    with pytest.raises(ValueError, match="static rigid mechanism"):
        _analysis(shell_supports=False)
    owner = _analysis()
    owner.shell_model.add_material("elastic", 2e5, 0.3, density=1.0)
    with pytest.raises(ValueError, match="model/contact definition changed"):
        owner._assemble(owner.initial.mechanical,
                        owner.initial.shell_displacement, 0.0,
                        owner.initial.native_histories)


def test_split_restart_replays_native_and_shell_accepted_state():
    targets = (0.5, 1.0)
    full = _analysis(curved=True, targets=targets).solve()
    assert full.status == "completed", full.failure
    prefix = _analysis(curved=True, targets=targets).solve(stop_after=1)
    assert prefix.status == "paused", prefix.failure
    assert prefix.accepted.completed_targets == 1
    resumed = _analysis(curved=True, targets=targets).solve(
        checkpoint=prefix.checkpoint,
        expected_sha256=prefix.checkpoint_sha256,
    )
    assert resumed.status == "completed", resumed.failure
    assert resumed.checkpoint == full.checkpoint
    np.testing.assert_array_equal(resumed.accepted.shell_displacement,
                                  full.accepted.shell_displacement)
    np.testing.assert_array_equal(resumed.accepted.mechanical.positions,
                                  full.accepted.mechanical.positions)
    assert sha(resumed.accepted.native_histories) == sha(full.accepted.native_histories)
    assert resumed.accepted.contact.contact.potential_j == full.accepted.contact.contact.potential_j


def test_wrong_model_and_rehashed_checkpoint_tampering_rejected():
    prefix = _analysis(targets=(0.5, 1.0)).solve(stop_after=1)
    assert prefix.status == "paused", prefix.failure
    with pytest.raises(ValueError, match="definition/settings binding"):
        _analysis(targets=(0.5, 1.0), shell_elastic_modulus=2e5).solve(
            checkpoint=prefix.checkpoint,
            expected_sha256=prefix.checkpoint_sha256,
        )
    broken = bytearray(prefix.checkpoint)
    broken[-2] = ord("0") if broken[-2] != ord("0") else ord("1")
    with pytest.raises(ValueError, match="external checkpoint hash mismatch"):
        _analysis(targets=(0.5, 1.0)).solve(
            checkpoint=bytes(broken),
            expected_sha256=prefix.checkpoint_sha256,
        )
    value = json.loads(prefix.checkpoint)
    value["records"][0]["contact"]["potential_j"] += 1e-6
    body = {key: item for key, item in value.items()
            if key != "checkpoint_sha256"}
    value["checkpoint_sha256"] = sha(body)
    forged = canonical(value)
    with pytest.raises(ValueError, match="physical replay mismatch"):
        _analysis(targets=(0.5, 1.0)).solve(
            checkpoint=forged,
            expected_sha256=sha256(forged).hexdigest(),
        )


def test_second_increment_failure_returns_accepted_checkpoint_prefix():
    owner = _analysis(targets=(0.0, 1.0), iterations=1)
    run = owner.solve()
    assert run.status == "failed"
    assert run.accepted.completed_targets == 1
    restored = _analysis(targets=(0.0, 1.0), iterations=1).solve(
        checkpoint=run.checkpoint,
        expected_sha256=run.checkpoint_sha256,
        stop_after=1,
    )
    assert restored.status == "paused", restored.failure
    assert restored.checkpoint == run.checkpoint
    np.testing.assert_array_equal(restored.accepted.shell_displacement,
                                  run.accepted.shell_displacement)
