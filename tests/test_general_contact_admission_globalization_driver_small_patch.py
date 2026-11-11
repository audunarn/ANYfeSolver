"""Bounded 4x4 Q4/S3 driver checks for the declared macro physical substeps.

Fresh driver-checks-v2 scope. The v5 driver module authored six real
caller/state/restart checks on the 16Q4 interior-press fixture but no
numerical allowance ever executed them. This module delivers those six
checks on the smallest physically meaningful caller fixture: a Q4
source patch against the family's two-S3 nonmatching target, with the
family's own perimeter support, interior distributed load shape,
materials, contact definition, settings and two-macro force-ramp
program. The family's build_case column gate does not admit two
columns, so the fixture is replicated here mechanic-for-mechanic
instead of edited in scripts/.

Discretization record (prior failure preserved, not claimed passed):
this fixture originally used a 2x2 Q4 source patch
(source_columns=2, source_rows=2, one interior load node). The
verified independent origin-linear discriminator
(smallpatch-linear-discriminator-v2) showed that the 2x2 fixture's
equilibrated half-load state violates the unchanged 50um gap budget
at the loaded point (loaded physical gap -122.672um); that failed
record stands, and no claim is made that the original 2x2 fixture
passed or that its exact nonlinear problem is infeasible. Root
adoption (2026-10-11), based on the verified independent
origin-linear results in
.development/contact-restart-20261011/smallpatch-resolution-v1
(corrected v2): the fixture discretization is now the prospective
4x4 Q4 resolution candidate - 25 source nodes, 16 Q4 elements, 25
explicit fine Q1 nodal pressure modes - with an explicit node remap
that keeps the physical load/anchor positions unchanged: load node
13 at the patch center (.6,.6) (NOT the grid-helper interior node
7), and ux anchors fine nodes 1 and 11 at (.1,.1) and (.1,.6) (NOT
the grid-helper node 6). At the half load -0.02N that candidate's
origin-linear equilibrated state satisfies exact KKT/wrench checks
to ~1e-16 with loaded physical gap -31.7757um inside the unchanged
50um budget (sampled field evidence, not a global certificate).
Only the spatial discretization and this remap change: g0=20um,
material E=1e5 nu=0.3 thickness 0.05, the two target S3 positions,
materials, clamps and offsets, source square (.1,1.1), point force
(.6,.6) with the full -0.04N two-step ramp, 50um gap budget,
tolerances, physics, control, solver choice and certificate are all
unchanged.

The fixture's actual load, pressure and solved active response are
explicit: the loaded interior node carries the full bounded press load
(0 N -> -0.02 N -> -0.04 N over the two macro steps), and every solved
reference asserted here must reach real active contact (active pressure
DOFs, positive numerical pressure, gap within the family tolerance).

The common reference slope is the family's flat 1e-4 variant (the
recorded family tractability run
d1_actual_cutback_single_press_20260929 used exactly this value): with
the 1e-2 default the tilted normal makes the projected gap bound's
sensitivity across a coarse 0.5 m 2x2 source face comparable to the
gap itself, which stagnates the swept certificate into the continuation
cutback on this fixture (observed with stack evidence in the
driver-checks-v2 phase probe). The flat slope is a family-admitted
fixture setting; no report/equilibrium/gap/NCP/raw-certificate gate is
disabled.

v7 near-contact declaration (parent instruction, not an accepted
engineering response; original 2x2 fixture rationale, geometry
retained unchanged for the 4x4 candidate): the caller fixture's
source reference height is 0.04002 (initial normal gap g0 = 20um)
instead of the family's 0.04004 (g0 = 40um). The v7 exact-rational
discriminator showed that on the original 2x2 fixture at g0=40um
the interior node's weak nodal-pressure onset (bilinear tent
weighting ratio 4/9) lands exactly at the -50um strong gap limit,
so weak pressure cannot activate inside the certificate's
admissible domain; at g0=20um the onset sits at -25um, safely
inside the unchanged 50um budget. Only this caller fixture's
discretization changes; material, load, physical work/state/gates,
active-contact assertions and all required checks are unchanged.

One declared injected first-refusal (contact_semismooth_stagnation on
the first macro attempt) is used to reach real continuation substeps.
Full and half references are shared immutable module caches; reference
results are restore/equality witnesses, not production-kernel scientific
oracles. The twenty-seven v5-verified math/public/caller checks are not
repeated; only the fixture-changed driver checks run here. An autouse
call-through phase logger records per-call certificate/report/increment
evidence into the attempt temp directory; it changes no production
behavior and starts no watchdog thread (a repeating faulthandler
dump thread killed this process with a Windows access violation
inside NumPy C reductions).
"""
import copy
import os
import tempfile
import time
from dataclasses import replace

import numpy as np
import pytest

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactPathControl, ContactPathNodalForce, ContactPathProgram,
    ContactSolverSettings, ContactSurface, create_shell_element,
    solve_static_nonlinear,
)
from anyfesolver.control import SolveCancelled
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact import s3_patch_static, static_path
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from anyfesolver.general_contact.semismooth_path import SemismoothStep
from anyfesolver.nonlinear_restart import (
    _CHECKPOINT_HASH_KEY,
    _checkpoint_without_hash,
    _sha256,
)

_CONTACT_PATH_PHYSICAL_SUBSTEPS_REVISION = "declared_macro_path_physical_substeps_v1"
_FORCE_N = 0.04
_GAP_TOLERANCE_M = 5.0e-5
_REFERENCE_SLOPE = 1.0e-4
_TARGET_RIGHT_X_M = 1.3
_SOURCE_COLUMNS = 4
_SOURCE_ROWS = 4
_LOAD_NODE = 13
_ANCHOR_NODES = (1, 11)


def _case():
    """Physical 4x4 Q4 press patch of the interior-press family.

    Replicates build_interior_press(source_support="perimeter",
    load_shape="interior_distributed", force_n=0.04) at
    source_columns=source_rows=4 (root-adopted resolution candidate):
    same material (E=1e5, nu=0.3), shell thickness 0.05, the family's
    flat 1e-4 common reference slope, full-width two-S3 target at z=0
    with reference surface offset 0.01, perimeter uz support with
    lateral uy guide, target left-edge control support, monolithic
    active-set enforcement with the D1 interior profile, and the
    family's two-step nodal force ramp on the loaded interior node.

    Explicit 4x4 node remap (root adoption; the grid-helper formulas
    1 + stride + 1 and columns + 2 would tie the load and the second
    anchor to the first interior fine nodes 7 and 6, which are NOT the
    original physical positions): the load is fine node 13 at the
    patch center (.6,.6) and the ux controls are fine nodes 1 and 11
    at (.1,.1) and (.1,.6) - the same physical positions as the
    original 2x2 fixture's load node 5 and control nodes 1 and 4. The
    remapped positions are asserted below against the actual mesh
    coordinates, not against grid indices.

    Declared near-contact caller fixture (v7 discriminator, parent
    instruction; not an accepted engineering response; original 2x2
    rationale, geometry retained unchanged): source reference height
    0.04002 instead of the family's 0.04004, same skins/target.
    Exact geometry: initial normal gap g0 = 0.04002 - 0.025 - 0.015
    = 20um. On the original 2x2 fixture the interior node's weak
    nodal-pressure onset (tent weighting ratio
    integral(phi^2)/integral(phi) = 4/9) sits at w = -45um with
    center physical gap -25um, safely inside the unchanged 50um gap
    budget; at the family's 40um the onset sits exactly at the -50um
    strong limit and cannot activate inside the admissible domain.
    """
    columns, rows = _SOURCE_COLUMNS, _SOURCE_ROWS
    stride = columns + 1
    model = FEModel(f"mixed_{columns}_q4_x_jet2_schur_press")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    common_normal = np.asarray((-_REFERENCE_SLOPE, 0.0, 1.0))
    common_normal /= np.linalg.norm(common_normal)
    source_reference_height = 0.04002/common_normal[2]
    source_y_knots = np.linspace(0.0, 1.0, rows + 1)
    for row in range(rows + 1):
        y = 0.1 + source_y_knots[row]
        for column in range(stride):
            x = 0.1 + column/columns
            model.add_node(1 + row*stride + column, x, y,
                           source_reference_height
                           + _REFERENCE_SLOPE*(x - 0.5*_TARGET_RIGHT_X_M))
    target_y0, target_y1 = -0.1, 1.2
    target_nodes = tuple(range((rows + 1)*stride + 1,
                               (rows + 1)*stride + 1 + 4))
    target_positions = ((0.0, target_y0, 0.0),
                        (_TARGET_RIGHT_X_M, target_y0, 0.0),
                        (_TARGET_RIGHT_X_M, target_y1, 0.0),
                        (0.0, target_y1, 0.0))
    for node, (x, y, z) in zip(target_nodes, target_positions):
        model.add_node(node, x, y,
                       z + _REFERENCE_SLOPE*(x - 0.5*_TARGET_RIGHT_X_M))
    target_normal = np.asarray((-_REFERENCE_SLOPE, 0.0, 1.0))
    target_normal /= np.linalg.norm(target_normal)
    for row in range(rows):
        for column in range(columns):
            element_id = 1 + row*columns + column
            bottom = 1 + row*stride + column
            model.add_element(element_id, create_shell_element(
                element_id,
                (bottom, bottom + 1, bottom + stride + 1, bottom + stride),
                "elastic", thickness=0.05,
                reference_normal=tuple(common_normal),
            ))
    target_faces = ((target_nodes[0], target_nodes[1], target_nodes[3]),
                    (target_nodes[1], target_nodes[2], target_nodes[3]))
    for element_id, nodes in enumerate(target_faces,
                                       start=columns*rows + 1):
        model.add_element(element_id, create_shell_element(
            element_id, nodes, "elastic", thickness=0.05,
            formulation="e4-pl-s3-v2d",
            reference_normal=tuple(target_normal),
            reference_surface_offset=0.01,
        ))
    model.add_boundary_condition(BoundaryCondition(
        "source_lateral_guide", tuple(range(1, target_nodes[0])), {"uy": 0.0}))
    rim = (tuple(range(1, stride + 1))
           + tuple(node for row in range(1, rows)
                   for node in (1 + row*stride, (row + 1)*stride))
           + tuple(range(1 + rows*stride, 1 + (rows + 1)*stride)))
    model.add_boundary_condition(BoundaryCondition(
        "source_perimeter_support", rim, {"uz": 0.0}))
    model.add_boundary_condition(BoundaryCondition(
        "target_control_support", (target_nodes[0], target_nodes[3]),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")}))
    definition = ContactDefinition(
        surfaces=(
            ContactSurface("q4_source", tuple(range(1, columns*rows + 1)),
                           "bottom"),
            ContactSurface("s3_target", (columns*rows + 1, columns*rows + 2),
                           "top"),
        ),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "pair", "q4_source", "s3_target", 10000.0, _GAP_TOLERANCE_M,
            normal_law="augmented"),),
    )
    settings = ContactSolverSettings(
        pair_motion="general", normal_discretization="primal_mortar",
        q4_source_gap_certificate="projected_matching",
        general_s3_difference_step=2.0e-5,
        maximum_general_s3_energy_evaluations=14144,
        maximum_candidate_pairs=max(32, columns*rows*len(target_faces)),
        q4_source_area_tolerance=1.0e-9,
        q4_source_weighted_gap_tolerance=1.0e-10,
        maximum_q4_source_integration_triangles=300,
        general_s3_differentiation="jet2_q4_s3_dev",
        q4_source_support_gradient_tolerance=1.0e-5,
        q4_source_support_hessian_tolerance=1.0e-5,
        q4_source_weighted_gap_gradient_tolerance=1.0e-5,
        q4_source_weighted_gap_hessian_tolerance=1.0e-5,
        pressure_tolerance=1.0e-5, maximum_augmentations=7,
        augmentation_acceleration="supported_schur_dev",
        augmentation_schur_maximum_modes=(rows + 1)*stride,
        augmentation_schur_maximum_pressure_pa=10.0,
        augmentation_schur_maximum_step_pa=5.0,
        augmentation_schur_difference_step_m=2.0e-6,
        augmentation_schur_maximum_attempts=2,
        augmentation_schur_maximum_relinearizations=0,
    )
    settings = replace(
        settings, enforcement_solver="monolithic_active_set",
        complementarity_force_scale_n=_FORCE_N,
        augmentation_acceleration="none",
        augmentation_schur_maximum_modes=None,
        augmentation_schur_maximum_pressure_pa=None,
        augmentation_schur_maximum_step_pa=None,
        augmentation_schur_difference_step_m=None,
        augmentation_schur_maximum_attempts=1,
        augmentation_schur_maximum_relinearizations=0,
        d1_interior_profile=True,
        d1_interior_coverage_tolerance_m2=1.0e-10,
        d1_interior_boundary_force_tolerance_n=1.0e-10,
    )
    # Explicit 4x4 remap (root adoption): the grid-helper formulas
    # 1 + stride + 1 and columns + 2 would tie the load and the
    # second anchor to the first interior fine nodes 7 and 6, which
    # are NOT the original physical positions. Keep the actual
    # physical positions: load node 13 at the patch center (.6,.6),
    # ux anchors nodes 1 and 11 at (.1,.1) and (.1,.6), asserted
    # against the mesh coordinates below.
    load_node = _LOAD_NODE
    anchors = _ANCHOR_NODES
    for node, expected_xy in ((load_node, (0.6, 0.6)),
                              (anchors[0], (0.1, 0.1)),
                              (anchors[1], (0.1, 0.6))):
        position = model.mesh.get_node(node)
        assert position is not None, f"remap node {node} missing"
        assert (abs(position.x - expected_xy[0]) < 1e-12
                and abs(position.y - expected_xy[1]) < 1e-12), (
            f"remap node {node} at ({position.x}, {position.y}), "
            f"expected physical position {expected_xy}")
    program = ContactPathProgram(
        controls=(ContactPathControl.node(anchors[0], "ux"),
                  ContactPathControl.node(anchors[1], "ux")),
        waypoints=((0.0, 0.0), (0.0, 0.0)), steps_per_segment=2,
        nodal_forces=(ContactPathNodalForce(
            load_node, "uz", (0.0, -_FORCE_N)),),
    )
    return model, definition, settings, program


def _enabled(settings):
    enabled = replace(settings, q4_source_gap_certificate="overlap_local_dev",
                      q4_source_overlap_dual_dev=True,
                      path_continuation_maximum_halvings=6)
    # v8 effective-settings defect repair: the prior enable selected the
    # overlap-local certificate and continuation but left the joint overlap
    # dual at its False core default, so the expensive driver attempts
    # exercised the old certificate instead of the already selected
    # joint-certificate candidate. Explicitly assert the returned effective
    # flag and that the intended D1/Jet2/fine-Q1/monolithic/primal-mortar
    # route is unchanged by the enable: this selects a tested algorithmic
    # candidate and changes no physics, tolerance or fixture setting.
    assert enabled.q4_source_overlap_dual_dev is True
    assert enabled.q4_source_gap_certificate == "overlap_local_dev"
    assert enabled.path_continuation_maximum_halvings == 6
    for name, intended in (("d1_interior_profile", True),
                           ("general_s3_differentiation", "jet2_q4_s3_dev"),
                           ("q4_source_pressure_basis", "fine_nodal_q1"),
                           ("normal_discretization", "primal_mortar"),
                           ("enforcement_solver", "monolithic_active_set"),
                           ("shell_skin_kinematics", "reference_linear"),
                           ("shell_pair_orientation", "legacy")):
        assert getattr(enabled, name) == intended == getattr(settings, name)
    return enabled


def _solve(model, definition, settings, program, **kwargs):
    return solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18, **kwargs)


_CACHE = {}


def _reference_full():
    if "full" not in _CACHE:
        model, definition, settings, program = _case()
        result = _solve(model, definition, _enabled(settings), program,
                        emit_restart_checkpoint=True)
        assert result.status == "completed" and len(result.steps) == 2
        checkpoint = result.to_restart_checkpoint()
        history = checkpoint["path_state"]["history"]
        assert len(history) == 2
        # A genuinely full macro attempt carries no continuation block.
        assert all("continuation" not in record for record in history)
        _CACHE["full"] = (result, checkpoint)
    return _CACHE["full"]


def _reference_half():
    if "half" not in _CACHE:
        model, definition, settings, program = _case()
        result = _solve(model, definition, _enabled(settings), program,
                        max_load_factor=0.5, emit_restart_checkpoint=True)
        assert result.status == "completed" and len(result.steps) == 1
        _CACHE["half"] = result
    return _CACHE["half"]


def _assert_active_contact(record):
    contact = record["contact"]
    assert contact["active_pressure_dofs"] >= 1
    assert max(contact["numerical_pressure_pa"]) > 0.0
    assert contact["minimum_gap"] <= _GAP_TOLERANCE_M


def _assert_matches_half(rerun):
    reference = _reference_half()
    assert rerun.status == "completed" and len(rerun.steps) == 1
    np.testing.assert_allclose(rerun.displacements, reference.displacements,
                               rtol=0., atol=1e-10)
    np.testing.assert_allclose(
        rerun.info["general_contact"]["integration_force"],
        reference.info["general_contact"]["integration_force"],
        rtol=0., atol=1e-10)
    rerun_checkpoint = rerun.to_restart_checkpoint()
    assert rerun_checkpoint == reference.to_restart_checkpoint()
    # The transaction rerun reaches the same real active contact state.
    _assert_active_contact(rerun_checkpoint["path_state"]["history"][0])


class _CancelAt:
    def __init__(self, stage):
        self.stage = stage

    def raise_if_cancelled(self, stage=""):
        if stage == self.stage:
            raise SolveCancelled("injected continuation cancellation",
                                 stage=stage)


def _force_first_macro_refusal(monkeypatch, calls):
    """One declared injected first-refusal to reach real substeps."""
    original = static_path.solve_semismooth_increment

    def wrapper(*args, **kwargs):
        calls.append({
            "target": np.asarray(kwargs["target"], dtype=float).copy(),
            "reduced_external":
                np.asarray(kwargs["reduced_external"], dtype=float).copy(),
            "iterations": None,
        })
        if len(calls) == 1:
            # Typed conformance: the production refusal path reads
            # step_result.initial_seed unconditionally, so the
            # injected refusal is a real SemismoothStep, not a loose
            # namespace mock.
            return SemismoothStep(
                completed=False, reason="contact_semismooth_stagnation",
                q=np.asarray(kwargs["q"], dtype=float).copy(),
                reactions=np.asarray(kwargs["reactions"], dtype=float).copy(),
                nodal_forces=np.asarray(kwargs["nodal_forces"], dtype=float).copy(),
                candidate_states=None, contact=None, iterations=3,
                residual_norm=float("inf"), constraint_norm=float("inf"),
                complementarity_norm=float("inf"), candidate_evaluations=2,
                backtracks=1)
        result = original(*args, **kwargs)
        calls[-1]["iterations"] = int(result.iterations)
        return result

    monkeypatch.setattr(static_path, "solve_semismooth_increment", wrapper)


_PHASE_LOG = None


def _phase_log():
    global _PHASE_LOG
    if _PHASE_LOG is None:
        _PHASE_LOG = open(os.path.join((os.environ.get("TEMP")
                                       or tempfile.gettempdir()),
                                       "small_patch_phase.log"),
                          "a", buffering=1)
    return _PHASE_LOG


@pytest.fixture(autouse=True)
def _phase_evidence(monkeypatch):
    """Call-through phase evidence; changes no production behavior."""
    log = _phase_log()
    log.write(f"{time.perf_counter():9.3f} phase fixture setup\n")

    def wrap(owner, name, label):
        original = getattr(owner, name)

        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                log.write(f"{time.perf_counter():9.3f} {label} done "
                           f"elapsed={time.perf_counter()-start:9.3f}\n")

        monkeypatch.setattr(owner, name, wrapper)

    wrap(s3_patch_static, "matching_projection_swept_lower", "matching")
    wrap(GeneralS3PatchManager, "evaluate", "evaluate")
    wrap(GeneralS3PatchManager, "admissible_step_fraction", "fraction")

    increment_original = static_path.solve_semismooth_increment

    def increment_wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = increment_original(*args, **kwargs)
        log.write(f"{time.perf_counter():9.3f} increment done "
                   f"elapsed={time.perf_counter()-start:9.3f} "
                   f"completed={result.completed} reason={result.reason} "
                   f"iterations={result.iterations}\n")
        return result

    monkeypatch.setattr(static_path, "solve_semismooth_increment",
                        increment_wrapper)
    # No repeating faulthandler watchdog here: with the watchdog
    # active, this process crashed with a Windows access violation
    # inside NumPy C reductions (observed in
    # q4_overlap_local_bound._mul/np.min on this exact fixture); the
    # crash was watchdog-correlated and the underlying race mechanism
    # is not independently established. The per-call phase log above
    # keeps the same timing evidence without a second thread.
    yield
    log.write(f"{time.perf_counter():9.3f} phase fixture teardown\n")


def test_small_patch_fixture_explicit_load_pressure_and_active_response():
    model, definition, settings, program = _case()
    # Explicit declared load: the loaded interior node carries the full
    # bounded press load, ramped by the program over two macro steps.
    # The assertions bind the actual physical load/anchor positions,
    # not the old 2x2 grid indices.
    assert program.nodal_forces[0].node_id == _LOAD_NODE
    loaded = model.mesh.get_node(_LOAD_NODE)
    assert abs(loaded.x - 0.6) < 1e-12 and abs(loaded.y - 0.6) < 1e-12
    assert [term[0] for control in program.controls
            for term in control.terms] == list(_ANCHOR_NODES)
    for node, expected_xy in zip(_ANCHOR_NODES, ((0.1, 0.1), (0.1, 0.6))):
        anchor = model.mesh.get_node(node)
        assert (abs(anchor.x - expected_xy[0]) < 1e-12
                and abs(anchor.y - expected_xy[1]) < 1e-12)
    assert program.nodal_forces[0].dof == "uz"
    np.testing.assert_allclose(program.force_values_at(0), [0.0],
                               rtol=0., atol=0.)
    np.testing.assert_allclose(program.force_values_at(1), [-_FORCE_N/2],
                               rtol=0., atol=0.)
    np.testing.assert_allclose(program.force_values_at(2), [-_FORCE_N],
                               rtol=0., atol=0.)
    result = _solve(model, definition, _enabled(settings), program,
                    max_load_factor=0.5, emit_restart_checkpoint=True)
    assert result.status == "completed" and len(result.steps) == 1
    checkpoint = result.to_restart_checkpoint()
    record = checkpoint["path_state"]["history"][0]
    # Explicit solved active response at the half load: real active
    # contact with positive pressure inside the family gap tolerance.
    _assert_active_contact(record)
    # Explicit interior deformation: the loaded interior node presses
    # toward the target while the perimeter support holds the rim.
    center_uz = model.mesh.dof_manager.get_node_dofs(
        program.nodal_forces[0].node_id)[2]
    assert result.displacements[center_uz] < 0.0
    _CACHE["half"] = result


def test_cancellation_during_substeps_restores_and_propagates(monkeypatch):
    _reference_half()
    model, definition, settings, program = _case()
    calls = []
    _force_first_macro_refusal(monkeypatch, calls)
    token = _CancelAt("contact_path.step:1.continuation:1")
    with pytest.raises(SolveCancelled,
                       match="injected continuation cancellation"):
        _solve(model, definition, _enabled(settings), program,
               cancellation_token=token)
    # The refused macro attempt published no substep Newton work.
    assert len(calls) == 1
    rerun = _solve(model, definition, _enabled(settings), program,
                   max_load_factor=0.5, emit_restart_checkpoint=True)
    _assert_matches_half(rerun)


def test_cancellation_inside_first_macro_attempt_restores_and_propagates():
    _reference_half()
    model, definition, settings, program = _case()
    token = _CancelAt("contact_path.step:1.semismooth.iteration:1")
    with pytest.raises(SolveCancelled,
                       match="injected continuation cancellation"):
        _solve(model, definition, _enabled(settings), program,
               cancellation_token=token)
    rerun = _solve(model, definition, _enabled(settings), program,
                   max_load_factor=0.5, emit_restart_checkpoint=True)
    _assert_matches_half(rerun)


def test_report_boundary_exception_restores_complete_macro_snapshot(monkeypatch):
    _reference_half()
    model, definition, settings, program = _case()
    original_increment = static_path.solve_semismooth_increment
    original_evaluate = GeneralS3PatchManager.evaluate
    armed = [False]
    raised = [False]

    def arming_increment(*args, **kwargs):
        result = original_increment(*args, **kwargs)
        armed[0] = True
        return result

    def raise_once(self, displacement, *, tangent=True):
        if armed[0] and not tangent and not raised[0]:
            raised[0] = True
            raise RuntimeError("injected report failure")
        return original_evaluate(self, displacement, tangent=tangent)

    monkeypatch.setattr(static_path, "solve_semismooth_increment",
                        arming_increment)
    monkeypatch.setattr(GeneralS3PatchManager, "evaluate", raise_once)
    with pytest.raises(RuntimeError, match="injected report failure"):
        _solve(model, definition, _enabled(settings), program)
    assert raised[0]
    rerun = _solve(model, definition, _enabled(settings), program,
                   max_load_factor=0.5, emit_restart_checkpoint=True)
    _assert_matches_half(rerun)


def test_post_commit_boundary_exception_restores_complete_macro_snapshot(monkeypatch):
    _reference_half()
    model, definition, settings, program = _case()
    original = static_path._support_reaction_resultants
    raised = [False]

    def raise_once(*args, **kwargs):
        if not raised[0]:
            raised[0] = True
            raise RuntimeError("injected post-commit failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(static_path, "_support_reaction_resultants",
                        raise_once)
    with pytest.raises(RuntimeError, match="injected post-commit failure"):
        _solve(model, definition, _enabled(settings), program)
    assert raised[0]
    rerun = _solve(model, definition, _enabled(settings), program,
                   max_load_factor=0.5, emit_restart_checkpoint=True)
    _assert_matches_half(rerun)


def test_cutback_chronology_force_ramp_and_gates(monkeypatch):
    reference, _checkpoint = _reference_full()
    model, definition, settings, program = _case()
    calls = []
    _force_first_macro_refusal(monkeypatch, calls)
    result = _solve(model, definition, _enabled(settings), program,
                    emit_restart_checkpoint=True)
    assert result.status == "completed" and len(result.steps) == 2
    np.testing.assert_allclose(result.displacements, reference.displacements,
                               rtol=0., atol=5e-8)
    np.testing.assert_allclose(
        result.info["general_contact"]["integration_force"],
        reference.info["general_contact"]["integration_force"],
        rtol=0., atol=5e-8)
    history = result.to_restart_checkpoint()["path_state"]["history"]
    assert len(history) == 2
    first, second = history
    # Active contact is actually reached on the cutback trajectory.
    _assert_active_contact(first)
    _assert_active_contact(second)
    continuation = first["continuation"]
    assert continuation["mode"] == "declared_physical_dyadic_substeps"
    assert (continuation["revision"]
            == _CONTACT_PATH_PHYSICAL_SUBSTEPS_REVISION)
    assert continuation["cutback_reason"] == "contact_semismooth_stagnation"
    assert continuation["maximum_halvings"] == 6
    assert "continuation" not in second
    substeps = continuation["substeps"]
    assert substeps
    tau = 0.0
    for substep in substeps:
        assert substep["tau_start"] == tau
        span = substep["tau_end"] - substep["tau_start"]
        scaled = span * float(1 << 7)
        assert scaled > 0.0 and scaled.is_integer()
        tau = substep["tau_end"]
    assert tau == 1.0
    # Every substep lies on the program's own dyadic target/force ramp.
    macro = calls[0]
    sub_calls = calls[1:1 + len(substeps)]
    assert len(sub_calls) == len(substeps)
    for substep, call in zip(substeps, sub_calls):
        np.testing.assert_allclose(
            call["reduced_external"],
            substep["tau_end"] * macro["reduced_external"],
            rtol=0., atol=1e-16)
        np.testing.assert_allclose(
            call["target"],
            (1. - substep["tau_end"]) * program.target_at(0)
            + substep["tau_end"] * program.target_at(1),
            rtol=0., atol=1e-15)
    assert (first["semismooth_iterations"]
            == 3 + sum(call["iterations"] for call in sub_calls))
    assert (first["external_nodal_work_scope"]
            == "continuation_substep_trapezoid_not_exact_path_integral")
    assert "external_nodal_work_trapezoid_j" in first
    np.testing.assert_allclose(first["target"], program.target_at(1),
                               rtol=0., atol=1e-12)
    assert first["nodal_forces_n"] == program.force_values_at(1).tolist()


def test_restart_rejects_tampered_and_incompatible_continuation_history(monkeypatch):
    model, definition, settings, program = _case()
    enabled = _enabled(settings)
    calls = []
    _force_first_macro_refusal(monkeypatch, calls)
    half = _solve(model, definition, enabled, program, max_load_factor=0.5,
                  emit_restart_checkpoint=True)
    assert half.status == "completed" and len(half.steps) == 1
    checkpoint = half.to_restart_checkpoint()
    record = checkpoint["path_state"]["history"][0]
    assert record["continuation"]["substeps"]

    def retamper(mutate):
        bad = copy.deepcopy(checkpoint)
        mutate(bad["path_state"]["history"][0]["continuation"])
        bad[_CHECKPOINT_HASH_KEY] = _sha256(_checkpoint_without_hash(bad))
        return bad

    fresh = _case()
    with pytest.raises(ValueError, match="chronology is invalid"):
        _solve(fresh[0], fresh[1], enabled, fresh[3],
               restart_checkpoint=retamper(
                   lambda cont: cont["substeps"][0].__setitem__(
                       "tau_end", 0.3)))
    fresh = _case()
    with pytest.raises(ValueError, match="continuation block is incompatible"):
        _solve(fresh[0], fresh[1], enabled, fresh[3],
               restart_checkpoint=retamper(
                   lambda cont: cont.__setitem__("revision", "forged_v0")))
    fresh = _case()
    with pytest.raises(ValueError, match="analysis contract is incompatible"):
        _solve(fresh[0], fresh[1], replace(enabled,
                                           path_continuation_maximum_halvings=0),
               fresh[3], restart_checkpoint=checkpoint)
    # A clean macro-only restart completes the trajectory consistently.
    fresh = _case()
    resumed = _solve(fresh[0], fresh[1], enabled, fresh[3],
                     restart_checkpoint=checkpoint,
                     emit_restart_checkpoint=True)
    assert resumed.status == "completed" and len(resumed.steps) == 2
    reference, _checkpoint = _reference_full()
    np.testing.assert_allclose(resumed.displacements, reference.displacements,
                               rtol=0., atol=5e-8)
    resumed_history = (resumed.to_restart_checkpoint()
                       ["path_state"]["history"])
    assert len(resumed_history) == 2
    assert "continuation" not in resumed_history[1]
