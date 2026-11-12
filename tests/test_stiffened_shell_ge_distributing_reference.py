"""Test-only full-retained KKT comparison of distinct reference attachment laws.

Same physical supports, material blocks and loads; no accuracy qualification.
The rigid law also locks nodal drill, whereas distribution transfers yaw through
in-plane force couples. No test helper is a product solve/state API.
"""
from dataclasses import replace
import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse import linalg as spla

from anyfesolver import b3_ge
from anyfesolver._b2_physical_linear_batch_dev import _accurate_sparse_residual
from anyfesolver._ge_beam3_generalized_static_boundary import spatial_jacobian
from anyfesolver._stiffened_shell_b2_nodal_footprint import reference_shell_matrix
from anyfesolver._stiffened_shell_distributing_footprint import DistributingFootprintDefinition
from anyfesolver._stiffened_shell_ge_distributing_reference import bind_reference
from anyfesolver._stiffened_shell_ge_experimental import (
    ExperimentalGEAnalysis, FixedCoordinate, PointConnection, Q4_ALGORITHMIC_PULLBACK_V1,
    _linear_equilibrium, _scaled_constraint_gap,
)
from anyfesolver._stiffened_shell_nodal_footprint import NodalRigidFootprint
from test_stiffened_shell_ge_experimental import _panel


def _definition():
    base = _panel()
    e, nu = base.shell.young_pa, base.shell.poisson
    g = e/(2*(1+nu))
    section = b3_ge.EllipsoidalGeneralizedSection(
        np.diag((e*.01, g*.008, g*.008, g*1.6666666666666667e-5,
                 e*8.333333333333333e-6, e*8.333333333333333e-6)), np.eye(6), 1e20, 1e9)
    beam = b3_ge.define_beam("b3-ge", 1, (101, 102, 103),
        np.array(((1., 0., .1), (2., 0., .1), (3., 0., .1))),
        np.tile(np.eye(3), (3, 1, 1)), section, np.eye(6))
    cells = ((1, 2, 6, 5), (2, 3, 7, 6), (3, 4, 8, 7))
    shell = replace(base.shell, node_ids=tuple(range(1, 9)),
        coordinates_m=tuple((x, y, 0.) for y in (-.5, .5) for x in (0., 1., 2., 3.)),
        q4_elements=cells)
    return replace(base, shell=shell, beams=(beam,), points=(PointConnection(3, 102),),
        fixed=tuple(FixedCoordinate("shell", n, a) for n in (1, 5) for a in range(6)),
        q4_pullback_policy=Q4_ALGORITHMIC_PULLBACK_V1)


def _selection():
    return DistributingFootprintDefinition("two-host-area", (2, 3), 102, (1., 3., -.5, .5))


def _multi_definition():
    """One 12-Q4 supported panel; the fixed pad spans eight actual hosts."""
    base = _definition()
    cells = tuple((row*7+i+1, row*7+i+2, (row+1)*7+i+2, (row+1)*7+i+1)
                  for row in range(2) for i in range(6))
    shell = replace(base.shell, node_ids=tuple(range(1, 22)),
        coordinates_m=tuple((i/2, -.5+row/2, 0.) for row in range(3) for i in range(7)),
        q4_elements=cells)
    return replace(base, shell=shell, points=(PointConnection(12, 102),),
                   fixed=tuple(FixedCoordinate("shell", n, axis)
                               for n in (1, 8, 15) for axis in range(6)))


def _multi_selection():
    return DistributingFootprintDefinition("eight-host-area", (3, 4, 5, 6, 9, 10, 11, 12),
                                          102, (1., 3., -.5, .5))


@pytest.fixture(scope="module")
def multi_bound():
    owner = ExperimentalGEAnalysis(_multi_definition())
    try:
        bound = bind_reference(owner, _multi_selection())
        yield owner, bound, bound.constraint_matrix(owner)
    finally:
        owner.close()


def _load(owner):
    load = np.zeros(owner._K.shape[0])
    load[6*owner._shell_index[2]+2] = -100.
    load[6*owner._shell_index[4]+2] = 100.
    slot = owner._shell_count+6*owner._beam_index[103]
    load[slot+1], load[slot+5] = 20., 3.
    return load


def _solve(owner, C, load):
    """Owner's general sparse mixed solve and bounded accurate refinement, test only."""
    free = owner._free
    A = sparse.bmat(((owner._K[free, :][:, free], C[:, free].T),
                     (C[:, free], None)), format="csc")
    rhs = np.r_[load[free], np.zeros(C.shape[0])]
    row_scale = np.asarray(np.abs(C)@owner._coordinate_scale).ravel()
    factor = spla.splu(A)
    solution = factor.solve(rhs)

    def refine(value):
        if not np.isfinite(value).all():
            raise RuntimeError("finite mixed reference solution required")
        defect = _accurate_sparse_residual(A, value[:, None], rhs[:, None])[:, 0]
        corrected = value+factor.solve(-defect)
        if not np.isfinite(corrected).all():
            raise RuntimeError("finite mixed reference correction required")
        return corrected

    solution = refine(refine(solution))
    for attempt in range(2):
        q = np.zeros(owner._K.shape[0])
        q[free] = solution[:len(free)]
        lam = solution[len(free):]
        equilibrium = owner._K@q+C.T@lam-load
        relative = _linear_equilibrium(equilibrium[free], load[free], owner._coordinate_scale[free])
        gap = _scaled_constraint_gap(C@q, row_scale)
        if np.isfinite(relative) and relative <= 1e-8 and gap <= 1e-9:
            break
        if attempt == 0:
            solution = refine(solution)
    else:
        raise RuntimeError("mixed reference physical gates failed")
    support = np.zeros_like(q)
    support[owner._fixed] = equilibrium[owner._fixed]
    return q, lam, support, A, rhs, row_scale


def _oracle(owner, A, rhs, row_scale):
    """Bounded independent dense LU, congruence-balanced without tangent averaging."""
    pose_scale = owner._coordinate_scale[owner._free]
    K_scaled = sparse.diags(pose_scale)@owner._K[owner._free, :][:, owner._free]@sparse.diags(pose_scale)
    work_scale = np.max(np.abs(K_scaled.data))
    scale = np.r_[pose_scale, work_scale/row_scale]
    dense = A.toarray()*scale[:, None]*scale[None, :]
    answer = scale*np.linalg.solve(dense, scale*rhs)
    q = np.zeros(owner._K.shape[0])
    q[owner._free] = answer[:len(owner._free)]
    return q, answer[len(owner._free):]


def _relative(a, b):
    return np.linalg.norm(a-b)/max(np.linalg.norm(a), np.linalg.norm(b), np.finfo(float).tiny)


@pytest.fixture(scope="module")
def reference_bundle():
    definition = _definition()
    owner = ExperimentalGEAnalysis(definition)
    rigid_definition = replace(definition, points=(), footprints=(NodalRigidFootprint(
        "same-two-host-rigid", (2, 3, 4, 6, 7, 8), 102, definition.shell.q4_elements[1:]),))
    rigid = ExperimentalGEAnalysis(rigid_definition)
    original_C, original_K, original_A = owner._C.copy(), owner._K.copy(), owner._A.copy()
    bound = bind_reference(owner, _selection())
    C = bound.constraint_matrix(owner)
    load = _load(owner)
    distributed_solution = _solve(owner, C, load)
    rigid_solution = _solve(owner, rigid._C, load)
    assert (owner._C != original_C).nnz == (owner._K != original_K).nnz == (owner._A != original_A).nnz == 0
    try:
        yield owner, rigid, bound, C, load, distributed_solution, rigid_solution
    finally:
        owner.close()
        rigid.close()


@pytest.fixture(scope="module")
def native_reference_tangents(reference_bundle):
    owner = reference_bundle[0]
    tangents = []
    for element, slots, origin in owner._beam_parts:
        frames = np.array([owner._beam_nodes[n][1] for n in element.node_ids])
        positions = np.array([owner._beam_nodes[n][0] for n in element.node_ids])
        native = element.operator.evaluate(positions, np.zeros((3, 3)), frames,
            np.tile(np.eye(3), (2, 1, 1)), np.zeros(18), origin=origin, check=element.operator.guard)
        tangent = spatial_jacobian(native.residual, native.hessian+native.hessian_low)
        tangents.append((owner._shell_count+slots, tangent))
    return tuple(tangents)


def test_actual_mapping_sparse_scatter_preserves_owner_and_all_retained_blocks(reference_bundle):
    owner, rigid, bound, C, load, *_ = reference_bundle
    assert C.shape == (6, owner._K.shape[0]) == (6, 90)
    assert C.has_canonical_format and np.isfinite(C.data).all() and np.all(C.data != 0.)
    np.testing.assert_array_equal(bound.constraint_scale, np.asarray(np.abs(C)@owner._coordinate_scale).ravel())
    assert len(bound.identity) == 64 and bound.identity != bound.model_identity
    assert bound.compiled.host_element_ids == (2, 3)
    assert set(bound.compiled.shell_node_ids) == {2, 3, 4, 6, 7, 8}
    assert bound.compiled.area_m2 == 2.
    np.testing.assert_array_equal(bound.compiled.centroid_m, [2., 0., 0.])
    np.testing.assert_array_equal(bound.compiled.eccentricity_m, [0., 0., .1])
    assert not set(bound.compiled.gather_slots).intersection(owner._fixed)
    assert C[:, owner._shell_count+6*len(owner._beam_ids):].nnz == 0
    assert rigid._C.shape == (36, 90)
    np.testing.assert_array_equal(owner._K.toarray(), rigid._K.toarray())
    np.testing.assert_array_equal(owner._fixed, rigid._fixed)
    assert owner._accepted is rigid._accepted is None
    assert owner._generation == rigid._generation == 0
    assert owner._linear_factor is rigid._linear_factor is None
    assert owner._C.shape == (6, 90)  # The original point operator is still present.
    assert (owner._C != C).nnz > 0
    full = np.arange(90.)/90.
    np.testing.assert_allclose(bound.gap(owner, full), C@full, rtol=0., atol=1e-15)
    assert np.count_nonzero(load) == 4


def test_multi_host_binding_scatter_owns_only_physical_selected_columns(multi_bound):
    owner, bound, matrix = multi_bound
    compiled = bound.compiled
    assert len(compiled.host_element_ids) == 8 and len(compiled.shell_node_ids) == 15
    assert compiled.constraint_matrix.shape == (6, 96)
    assert matrix.shape == (6, 168) and matrix.has_canonical_format
    assert sparse.isspmatrix_csr(matrix) and np.all(matrix.data != 0.)
    assert not set(compiled.gather_slots).intersection(owner._fixed)
    physical = owner._shell_count+6*len(owner._beam_ids)
    assert matrix[:, physical:].nnz == 0
    np.testing.assert_array_equal(matrix[:, compiled.gather_slots].toarray(), compiled.constraint_matrix)
    outside = np.setdiff1d(np.arange(matrix.shape[1]), compiled.gather_slots)
    assert matrix[:, outside].nnz == 0
    full = np.sin(np.arange(matrix.shape[1])+.3)
    np.testing.assert_allclose(bound.gap(owner, full), matrix@full, rtol=0., atol=2e-15)
    assert owner._generation == 0 and owner._accepted is None and owner._linear_factor is None


@pytest.mark.parametrize("axis", range(6))
def test_multi_host_bound_recovery_preserves_six_side_work_channels(multi_bound, axis):
    owner, bound, matrix = multi_bound
    multiplier = np.eye(6)[axis]
    transfer = bound.side_actions(owner, multiplier)
    assert transfer.shell_covectors.shape == (15, 6)
    np.testing.assert_array_equal(transfer.beam_covector, multiplier)
    np.testing.assert_allclose(transfer.shell_force, -multiplier[:3], rtol=0., atol=2e-14)
    np.testing.assert_allclose(transfer.shell_moment_about_centroid,
        -multiplier[3:]-np.cross(bound.compiled.eccentricity_m, multiplier[:3]), rtol=0., atol=2e-14)
    q = np.sin(np.arange(owner._K.shape[0])+.2)
    local = bound.compiled.gather(q).reshape(-1, 6)
    work = np.sum(transfer.shell_covectors*local[:-1])+transfer.beam_covector@local[-1]
    assert work == pytest.approx(multiplier@(matrix@q), abs=2e-14)
    np.testing.assert_array_equal(transfer.shell_covectors[:, 5], np.zeros(15))


def test_multi_host_binding_checks_last_shell_and_beam_offsets(multi_bound):
    owner, bound, _ = multi_bound
    shell = bound.compiled.shell_node_ids[-1]
    original = owner._shell_index[shell]
    try:
        owner._shell_index[shell] = original+1
        with pytest.raises(ValueError, match="stale local shell"):
            bound.gap(owner, np.zeros(owner._K.shape[0]))
    finally:
        owner._shell_index[shell] = original
    beam = bound.compiled.beam_node_id
    original = owner._beam_index[beam]
    try:
        owner._beam_index[beam] = original+1
        with pytest.raises(ValueError, match="stale physical beam"):
            bound.side_actions(owner, np.zeros(6))
    finally:
        owner._beam_index[beam] = original


def test_reference_point_coordinate_pullback_matches_actual_identity_frame_joint(reference_bundle):
    owner, *_ = reference_bundle
    joint = owner._connections[0]
    trial = joint.evaluate(joint.reference_positions, joint.reference_frames, np.zeros(6))
    expected = np.column_stack((reference_shell_matrix(*joint.reference_positions), np.eye(6)))
    np.testing.assert_allclose(trial.constraint_jacobian, expected, rtol=0., atol=1e-14)
    slots = owner._joint_slots[0]
    np.testing.assert_allclose(owner._C[:, slots].toarray(), expected, rtol=0., atol=1e-14)


@pytest.mark.parametrize("which", (0, 1))
def test_sparse_dense_full_mixed_response_reactions_energy_and_retained_compatibility(
        reference_bundle, native_reference_tangents, which):
    owner, rigid, bound, C, load, distributed, rigid_result = reference_bundle
    matrix = C if which == 0 else rigid._C
    q, lam, support, A, rhs, row_scale = (distributed, rigid_result)[which]
    dense_q, dense_lam = _oracle(owner, A, rhs, row_scale)
    scale = owner._coordinate_scale
    assert _relative(q/scale, dense_q/scale) <= 1e-10
    physical = owner._shell_count+6*len(owner._beam_ids)
    assert _relative(q[:physical]/scale[:physical], dense_q[:physical]/scale[:physical]) <= 1e-10
    assert _relative(lam*row_scale, dense_lam*row_scale) <= 1e-10
    dense_equilibrium = owner._K@dense_q+matrix.T@dense_lam-load
    # Independently balanced coordinates must satisfy the original raw operator.
    raw_dense = np.r_[dense_q[owner._free], dense_lam]
    raw_defect = _accurate_sparse_residual(A, raw_dense[:, None], rhs[:, None])[:, 0]
    assert _linear_equilibrium(raw_defect[:len(owner._free)], load[owner._free],
                               scale[owner._free]) <= 1e-8
    assert _scaled_constraint_gap(raw_defect[len(owner._free):], row_scale) <= 1e-9
    assert _relative(support[owner._fixed]*scale[owner._fixed],
                     dense_equilibrium[owner._fixed]*scale[owner._fixed]) <= 1e-10
    assert _linear_equilibrium((owner._K@q+matrix.T@lam-load)[owner._free],
                               load[owner._free], scale[owner._free]) <= 1e-8
    assert _scaled_constraint_gap(matrix@q, row_scale) <= 1e-9
    assert np.linalg.norm(q[physical:]) > 0.  # Twenty-four native variables remain solved.
    for global_slots, tangent in native_reference_tangents:
        retained_defect = (tangent@q[global_slots])[18:]
        assert np.linalg.norm(retained_defect*scale[global_slots[18:]])/np.linalg.norm(load*scale) <= 1e-8
    internal_work = float(q@(owner._K@q))
    external_work = float(q@load)
    assert internal_work == pytest.approx(external_work, rel=1e-10)
    assert .5*internal_work > 0.
    evidence = os.environ.get("R8_DISTRIBUTING_GE_EVIDENCE")
    if evidence:
        record = dict(law="distributing" if which == 0 else "rigid", full_coordinates=len(q),
            retained_coordinates=len(q)-physical, constraint_rows=matrix.shape[0],
            kkt_dimension=A.shape[0], kkt_nnz=A.nnz, constraint_nnz=matrix.nnz,
            scaled_sparse_dense_q=_relative(q/scale, dense_q/scale),
            scaled_sparse_dense_lambda=_relative(lam*row_scale, dense_lam*row_scale),
            scaled_equilibrium=_linear_equilibrium(
                (owner._K@q+matrix.T@lam-load)[owner._free], load[owner._free], scale[owner._free]),
            scaled_gap=_scaled_constraint_gap(matrix@q, row_scale),
            internal_work=internal_work, external_work=external_work,
            displacement=q.tolist(), multipliers=lam.tolist(), support_covectors=support.tolist())
        (Path(evidence)/f"{record['law']}-response-v1.json").write_text(
            json.dumps(record, indent=2, allow_nan=False)+"\n", encoding="utf-8")


@pytest.mark.parametrize("which", (0, 1))
def test_global_and_interface_force_moment_and_virtual_work(reference_bundle, which):
    owner, rigid, bound, C, load, distributed, rigid_result = reference_bundle
    q, lam, support, *_ = (distributed, rigid_result)[which]
    matrix = C if which == 0 else rigid._C
    physical = owner._shell_count+6*len(owner._beam_ids)
    positions = np.vstack((owner._shell_xyz, [owner._beam_nodes[n][0] for n in owner._beam_ids]))
    external = (load+support)[:physical].reshape(-1, 6)
    np.testing.assert_allclose(external[:, :3].sum(axis=0), 0., atol=1e-7)
    np.testing.assert_allclose((np.cross(positions, external[:, :3])+external[:, 3:]).sum(axis=0),
                               0., atol=1e-7)
    covectors = np.asarray(matrix.T@lam)[:physical].reshape(-1, 6)
    np.testing.assert_allclose(covectors[:, :3].sum(axis=0), 0., atol=1e-10)
    np.testing.assert_allclose((np.cross(positions, covectors[:, :3])+covectors[:, 3:]).sum(axis=0),
                               0., atol=1e-10)
    virtual = np.sin(np.arange(owner._K.shape[0])+.3)
    assert virtual@(matrix.T@lam) == pytest.approx(lam@(matrix@virtual), rel=1e-13, abs=1e-11)
    if which == 0:
        sides = bound.side_actions(owner, lam)
        np.testing.assert_array_equal(sides.beam_covector, lam)
        np.testing.assert_allclose(sides.shell_force, -lam[:3], atol=1e-10)
        np.testing.assert_allclose(sides.shell_moment_about_centroid,
                                  -lam[3:]-np.cross(sides.eccentricity_m, lam[:3]), atol=1e-10)
        gathered = virtual[bound.compiled.gather_slots].reshape(7, 6)
        assert np.sum(gathered[:6]*sides.shell_covectors)+gathered[6]@sides.beam_covector == pytest.approx(
            lam@bound.gap(owner, virtual), rel=1e-13, abs=1e-11)


def test_distribution_admits_nonrigid_slope_and_gives_different_coupled_flexibility(reference_bundle):
    owner, rigid, bound, C, load, distributed, rigid_result = reference_bundle
    mode = np.zeros(owner._K.shape[0])
    for node in bound.compiled.shell_node_ids:
        slot = 6*owner._shell_index[node]
        mode[slot+2] = owner._shell_xyz[owner._shell_index[node], 0]-2.
    np.testing.assert_allclose(C@mode, 0., atol=1e-15)
    assert np.linalg.norm(rigid._C@mode) > 1.
    assert mode@load > 0.
    q_distributed, q_rigid = distributed[0], rigid_result[0]
    np.testing.assert_allclose(C@q_rigid, 0., atol=1e-12)
    flexibility = float(load@q_distributed)
    rigid_flexibility = float(load@q_rigid)
    assert flexibility > rigid_flexibility*(1.+1e-6)


def test_sparse_buffers_and_side_results_are_owned(reference_bundle):
    owner, _, bound, _, _, distributed, _ = reference_bundle
    C = bound.constraint_matrix(owner)
    pristine = C.copy()
    for array in (C.data, C.indices, C.indptr):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(ValueError):
        bound.constraint_scale.setflags(write=True)
    C.data = np.zeros_like(C.data)  # Replacing a returned sparse object's buffer cannot change the binding.
    assert (bound.constraint_matrix(owner) != pristine).nnz == 0
    multiplier = distributed[1].copy()
    sides = bound.side_actions(owner, multiplier)
    multiplier[:] = 0.
    assert np.linalg.norm(sides.beam_covector) > 0.
    for array in (sides.shell_covectors, sides.beam_covector, sides.shell_force,
                  sides.shell_moment_about_centroid, sides.centroid_m, sides.eccentricity_m):
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_reordered_owner_nodes_and_host_identity_mapping_preserve_physical_scatter(reference_bundle):
    first, _, original, C, *_ = reference_bundle
    definition = first._definition
    order = tuple(reversed(range(8)))
    shell = replace(definition.shell, node_ids=tuple(definition.shell.node_ids[i] for i in order),
        coordinates_m=tuple(definition.shell.coordinates_m[i] for i in order),
        q4_elements=tuple(reversed(definition.shell.q4_elements)))
    owner = ExperimentalGEAnalysis(replace(definition, shell=shell))
    try:
        selection = replace(_selection(), host_element_ids=(1, 2))
        rebound = bind_reference(owner, selection)
        perm = [6*owner._shell_index[n]+a for n in first._shell_ids for a in range(6)]
        perm += list(range(owner._shell_count, owner._K.shape[0]))
        np.testing.assert_allclose(rebound.constraint_matrix(owner)[:, perm].toarray(), C.toarray(), rtol=0., atol=0.)
        assert original.model_identity != rebound.model_identity
        assert original.identity != rebound.identity
        assert original.compiled.identity != rebound.compiled.identity
    finally:
        owner.close()


def test_wrong_owner_and_unsupported_reference_policies_reject(reference_bundle):
    owner, rigid, bound, *_ = reference_bundle
    with pytest.raises(ValueError, match="another"):
        bound.constraint_matrix(rigid)
    with pytest.raises(ValueError, match="typed"):
        bind_reference(owner, {"hosts": (2, 3)})
    with pytest.raises(ValueError, match="element IDs"):
        bind_reference(owner, replace(_selection(), host_element_ids=(2, 99)))
    with pytest.raises(ValueError, match="beam node"):
        bind_reference(owner, replace(_selection(), beam_node_id=999))
    with pytest.raises(ValueError, match="dimension"):
        bound.gap(owner, np.zeros(owner._K.shape[0]-1))
    outside = np.zeros(owner._K.shape[0])
    outside[owner._fixed[0]] = np.nan
    with pytest.raises(ValueError, match="finite real full"):
        bound.gap(owner, outside)


def test_closed_stale_and_unsupported_owner_rejections_are_isolated():
    owner = ExperimentalGEAnalysis(_definition())
    try:
        bound = bind_reference(owner, _selection())
        identity = owner._identity
        owner._identity = "stale-model"
        with pytest.raises(ValueError, match="stale"):
            bound.constraint_matrix(owner)
        owner._identity = identity
        owner._generation = 1
        with pytest.raises(ValueError, match="stale"):
            bound.side_actions(owner, np.zeros(6))
        owner._generation = 0
        original_K = owner._K
        owner._K = owner._K[:, :-1]
        with pytest.raises(ValueError, match="stale"):
            bound.constraint_matrix(owner)
        owner._K = original_K
        original_index = owner._shell_index[2]
        owner._shell_index[2] = original_index+1
        with pytest.raises(ValueError, match="stale local shell"):
            bound.gap(owner, np.zeros(90))
        owner._shell_index[2] = original_index
        original_free = owner._free
        owner._free = np.setdiff1d(original_free, bound.compiled.gather_slots)
        with pytest.raises(ValueError, match="independent supported"):
            bind_reference(owner, _selection())
        owner._free = original_free
        original_definition = owner._definition
        for bad in (replace(original_definition, q4_pullback_policy="historical_corotational"),
                    replace(original_definition, bl5_material_policy="not-elastic"),
                    replace(original_definition, shell_reference_policy="not-flat"),
                    replace(original_definition, shell=replace(original_definition.shell,
                        s3_elements=((1, 2, 6),)))):
            owner._definition = bad
            with pytest.raises(ValueError, match="flat elastic"):
                bind_reference(owner, _selection())
        owner._definition = original_definition
        owner._finite_driver = object()
        with pytest.raises(ValueError, match="unadvanced"):
            bind_reference(owner, _selection())
        owner._finite_driver = None
        position, frame = owner._beam_nodes[102]
        owner._beam_nodes[102] = (position, np.diag((1., -1., -1.)))
        with pytest.raises(ValueError, match="identity"):
            bind_reference(owner, _selection())
        owner._beam_nodes[102] = (position, frame)
        owner.close()
        with pytest.raises(RuntimeError, match="closed"):
            bound.constraint_matrix(owner)
        with pytest.raises(RuntimeError, match="closed"):
            bind_reference(owner, _selection())
    finally:
        owner.close()
