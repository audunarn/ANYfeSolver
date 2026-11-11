"""Usable experimental reference-linear owner boundary; no accuracy qualification.

Reuse the proven three-Q4/one-B3 reference data and independent candidate solve.
No new solver or repeated dense/native oracle preparation is introduced here.
"""
from dataclasses import replace
import json

import numpy as np
import pytest
from scipy import sparse

from anyfesolver import stiffened_shell as api
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver._stiffened_shell_ge_checkpoint_binary import MAGIC, CHART_MAGIC, S3_MAGIC, encode
from anyfesolver._stiffened_shell_ge_experimental import CHECKPOINT, FINITE_CHECKPOINT
from anyfesolver._stiffened_shell_ge_distributing_reference import bind_reference
from test_stiffened_shell_ge_distributing_reference import (
    _definition, _selection, _load, _solve, _relative, _multi_definition, _multi_selection,
)


def _actions():
    return (api.NodalAction("shell", 2, (0., 0., -100., 0., 0., 0.)),
            api.NodalAction("shell", 4, (0., 0., 100., 0., 0., 0.)),
            api.NodalAction("beam", 103, (0., 20., 0., 0., 0., 3.)))


def _q(result):
    return np.r_[result.shell_displacement, result.beam_coordinates]


@pytest.fixture(scope="module")
def prepared():
    base = _definition()
    legacy = api.create_analysis(definition=base)
    product = mixed = None
    try:
        proof = bind_reference(legacy, _selection())
        candidate_C = proof.constraint_matrix(legacy)
        candidate = _solve(legacy, candidate_C, _load(legacy))
        definition = replace(base, points=(), distributing_footprints=(_selection(),))
        product = api.create_analysis(definition=definition)
        first = product.solve_linear(_actions())
        second = product.solve_linear(_actions())
        mixed_definition = replace(definition, points=(api.PointConnection(1, 101),))
        mixed = api.create_analysis(definition=mixed_definition)
        mixed_result = mixed.solve_linear(_actions())
        mixed_C = sparse.vstack((mixed._C[:6], candidate_C), format="csr")
        mixed_candidate = _solve(legacy, mixed_C, _load(legacy))
        yield dict(base=base, legacy=legacy, owner=product, definition=definition,
                   first=first, current=second, proof=proof, candidate_C=candidate_C,
                   candidate=candidate, mixed=mixed, mixed_result=mixed_result,
                   mixed_candidate=mixed_candidate)
    finally:
        if mixed is not None: mixed.close()
        if product is not None: product.close()
        legacy.close()


def test_explicit_facade_exports_and_narrow_capabilities(prepared):
    for name in ("GEReferenceDistributingFootprint", "GEReferenceDistributingFootprintSelection",
                 "GEReferenceDistributingFootprintWrench", "GE_DISTRIBUTING_REFERENCE_LINEAR_V1_EXPERIMENTAL"):
        assert name in api.__all__ and hasattr(api, name)
    owner = prepared["owner"]
    assert type(owner._definition.distributing_footprints[0]) is api.GEReferenceDistributingFootprint
    assert owner.capabilities["analysis"] == ("reference_linear",)
    assert owner.capabilities["loads"] == ("nodal dead actions",)
    assert owner.capabilities["checkpoints"] == ("reference_linear",)
    assert not owner.reference_modal_capabilities["eligible"]
    assert owner.preflight()["retained_dofs"] == 24
    assert owner._C.shape == (6, 90) and not owner._connections
    assert owner._identity != prepared["legacy"]._identity


@pytest.mark.parametrize("mixed", (False, True))
def test_public_response_matches_prior_independent_candidate_with_point_prefix(prepared, mixed):
    owner = prepared["mixed"] if mixed else prepared["owner"]
    result = prepared["mixed_result"] if mixed else prepared["first"]
    q, lam, support, *_ = prepared["mixed_candidate"] if mixed else prepared["candidate"]
    assert _relative(_q(result)/owner._coordinate_scale, q/owner._coordinate_scale) <= 1e-10
    assert _relative(result.multipliers*owner._constraint_scale, lam*owner._constraint_scale) <= 1e-10
    assert _relative(result.reactions[owner._fixed]*owner._coordinate_scale[owner._fixed],
                     support[owner._fixed]*owner._coordinate_scale[owner._fixed]) <= 1e-10
    np.testing.assert_array_equal(owner._K.toarray(), prepared["legacy"]._K.toarray())
    assert len(result.point_wrenches) == int(mixed)
    assert len(result.distributing_footprint_wrenches) == 1
    assert owner._distributing_bindings[0][1] == (6 if mixed else 0)
    if mixed:
        point = result.point_wrenches[0]
        assert (point.shell_node_id, point.beam_node_id) == (1, 101)
        resisting = owner._C[:6, owner._joint_slots[0]].T@result.multipliers[:6]
        np.testing.assert_allclose(np.r_[point.shell_action, point.beam_action], -resisting, atol=1e-12)
    else:
        assert (owner._C != prepared["candidate_C"]).nnz == 0
    physical = owner._shell_count+6*len(owner._beam_ids)
    assert owner._C[:, physical:].nnz == 0 and np.linalg.norm(_q(result)[physical:]) > 0.


@pytest.mark.parametrize("mixed", (False, True))
def test_physical_exerted_wrenches_work_selection_and_immutable_records(prepared, mixed):
    owner = prepared["mixed"] if mixed else prepared["owner"]
    result = prepared["mixed_result"] if mixed else prepared["current"]
    record, = owner.recover(selection=api.GEReferenceDistributingFootprintSelection(("two-host-area",)))
    assert type(record) is api.GEReferenceDistributingFootprintWrench
    assert record.model_identity == owner._identity and record.generation == result.generation
    assert record.law == api.GE_DISTRIBUTING_REFERENCE_LINEAR_V1_EXPERIMENTAL
    assert record.configuration == "reference_linear" and not record.production_qualified
    compiled, start = owner._distributing_bindings[0]
    lam = result.multipliers[start:start+6]
    np.testing.assert_allclose(record.beam_action, -lam, rtol=0., atol=0.)
    np.testing.assert_allclose(record.shell_action[:3], lam[:3], atol=1e-12)
    np.testing.assert_allclose(record.shell_action[3:], lam[3:]+np.cross(record.eccentricity_m, lam[:3]), atol=1e-12)
    np.testing.assert_array_equal(record.shell_node_actions[:, 5], np.zeros(6))
    virtual = np.sin(np.arange(owner._K.shape[0])+.2)
    local = virtual[compiled.gather_slots].reshape(7, 6)
    physical_work = np.sum(local[:6]*record.shell_node_actions)+local[6]@record.beam_action
    assert physical_work == pytest.approx(-lam@(owner._C[start:start+6]@virtual), abs=1e-10)
    for array in (record.shell_node_actions, record.shell_action, record.beam_action,
                  record.centroid_m, record.beam_position_m, record.eccentricity_m, record.gap):
        with pytest.raises(ValueError): array.setflags(write=True)
    with pytest.raises(ValueError, match="unknown"):
        owner.recover(selection=api.GEReferenceDistributingFootprintSelection(("missing",)))


def test_one_cached_factor_repeated_loads_and_cancelled_publication_keep_prefix(prepared):
    owner = prepared["owner"]
    assert owner.preflight()["reference_linear_factorizations"] == 1
    assert prepared["first"].generation == 1 and prepared["current"].generation == 2
    np.testing.assert_array_equal(_q(prepared["first"]), _q(prepared["current"]))
    before, generation = owner.checkpoint(), owner._generation
    class AtPublish(CancellationToken):
        def raise_if_cancelled(self, stage=""):
            if stage == "stiffened-shell.linear.publish": self.cancel("owned test boundary")
            super().raise_if_cancelled(stage)
    changed = tuple(replace(a, wrench=tuple(2*v for v in a.wrench)) for a in _actions())
    with pytest.raises(SolveCancelled):
        owner.solve_linear(changed, cancellation_token=AtPublish())
    assert owner._generation == generation and owner.checkpoint() == before
    assert owner._finite_driver is None and owner.preflight()["reference_linear_factorizations"] == 1


@pytest.mark.parametrize("binary", (False, True))
def test_fresh_checkpoint_validation_roundtrip_and_resume(prepared, binary):
    source = prepared["owner"]
    raw = source.checkpoint()
    value = json.loads(raw)
    fresh = api.create_analysis(definition=prepared["definition"])
    try:
        bad_schema = dict(value, schema=FINITE_CHECKPOINT)
        wrong_model = dict(value, model=prepared["legacy"]._identity)
        corrupted = json.loads(raw)
        corrupted["q"][fresh._fixed[0]] = .01
        retained_load = json.loads(raw)
        retained_load["load"][fresh._shell_count+6*len(fresh._beam_ids)] = 1.
        finite_binary = encode("finite", source._identity, value["generation"], ())
        for bad in (json.dumps(bad_schema).encode(), json.dumps(wrong_model).encode(),
                    json.dumps(corrupted).encode(), json.dumps(retained_load).encode(), finite_binary,
                    CHART_MAGIC+b"unsupported", S3_MAGIC+b"unsupported", b"ANYSGE_BODY1\0unsupported",
                    b"["*1100+b"0"+b"]"*1100):
            with pytest.raises(ValueError): fresh.restore(bad)
            assert fresh._accepted is None and fresh._generation == 0 and fresh._finite_driver is None
        arrays = tuple((name, np.array(value[name])) for name in ("q", "multipliers", "load"))
        saved = encode("linear", source._identity, value["generation"], arrays) if binary else raw
        if binary:
            corrupt_bytes = saved[:-1]+bytes((saved[-1]^1,))
            with pytest.raises(ValueError): fresh.restore(corrupt_bytes)
        restored = fresh.restore(saved)
        assert fresh.checkpoint() == raw
        np.testing.assert_array_equal(_q(restored), _q(source.recover()))
        record, = fresh.recover(selection=api.GEReferenceDistributingFootprintSelection(("two-host-area",)))
        np.testing.assert_array_equal(record.beam_action, source.recover().distributing_footprint_wrenches[0].beam_action)
        resumed = fresh.solve_linear(_actions())
        assert resumed.generation == restored.generation+1
        assert fresh.preflight()["reference_linear_factorizations"] == 1
        np.testing.assert_array_equal(_q(resumed), _q(restored))
    finally:
        fresh.close()


def test_legacy_empty_field_identity_capabilities_and_checkpoint_parity(prepared):
    old = prepared["legacy"]
    explicit_empty = api.create_analysis(definition=replace(prepared["base"], distributing_footprints=()))
    try:
        assert old._identity == explicit_empty._identity
        assert old.capabilities == explicit_empty.capabilities
        old_result = old.solve_linear(_actions())
        new_result = explicit_empty.solve_linear(_actions())
        assert old_result.distributing_footprint_wrenches == new_result.distributing_footprint_wrenches == ()
        assert old.checkpoint() == explicit_empty.checkpoint()
    finally:
        explicit_empty.close()


def test_owner_methods_and_common_load_spectral_admission_reject_before_work(prepared):
    from anyfesolver._stiffened_shell_ge_reference_modal import _admit
    from anyfesolver._stiffened_shell_ge_body_load import admits_owner as admits_body
    from anyfesolver._stiffened_shell_ge_distributed_load import admits_owner as admits_distributed
    owner = prepared["mixed"]  # Point presence must not reopen unverified wrappers.
    before = owner.checkpoint()
    for call in (lambda: owner.solve(_actions()),
                 lambda: owner.solve(_actions(), displacement_control=object()),
                 lambda: owner.solve(_actions(), increment_control=object()),
                 lambda: owner.rebase_rotations(owner._generation),
                 lambda: owner.continue_arc_length(_actions(), object()),
                 lambda: owner.modes(api.GEReferenceModesRequest()), lambda: _admit(owner)):
        with pytest.raises(ValueError, match="distributing"): call()
    wrappers = (api.GEBodyLoad(owner._identity, (0., 0., -9.81)),
                api.GEBeamDistributedLoad(owner._identity, (api.GEBeamLineAction(1, (0., 0., 1.)),)),
                api.GEShellPressureLoad(owner._identity, (api.GEShellPressureAction(2, 1.),)))
    assert not admits_body(owner) and not admits_distributed(owner)
    for request in wrappers:
        with pytest.raises(ValueError): request.compile(owner)
        with pytest.raises(ValueError, match="nodal dead"): owner.solve_linear(request)
    assert owner.checkpoint() == before and owner._finite_driver is None


def test_owned_caller_definition_geometry_and_action_inputs(prepared):
    ids, bounds = [2, 3], [1., 3., -.5, .5]
    group = api.GEReferenceDistributingFootprint("owned", ids, 102, bounds)
    ids[:] = [1, 99]; bounds[0] = 100.
    assert group.host_element_ids == (2, 3) and group.bounds_xy_m == (1., 3., -.5, .5)
    coordinates = [list(x) for x in prepared["base"].shell.coordinates_m]
    cells = [list(x) for x in prepared["base"].shell.q4_elements]
    definition = replace(prepared["base"], points=(), shell=replace(prepared["base"].shell,
        coordinates_m=coordinates, q4_elements=cells), distributing_footprints=(group,))
    owner = api.create_analysis(definition=definition)
    try:
        before = owner._identity
        coordinates[0][0] = 100.; cells[1][0] = 99
        wrench = np.array((0., 20., 0., 0., 0., 3.))
        result = owner.solve_linear((api.NodalAction("beam", 103, wrench),))
        wrench[:] = np.nan
        assert owner._identity == before and np.isfinite(_q(result)).all()
        assert owner._definition.shell.q4_elements[1] == (2, 3, 7, 6)
        with pytest.raises(ValueError, match="finite real"):
            owner.solve_linear((api.NodalAction("beam", 103, np.ones(6, dtype=complex)),))
    finally:
        owner.close()


@pytest.mark.parametrize("change", (
    lambda d: replace(d, distributing_footprints=list(d.distributing_footprints)),
    lambda d: replace(d, distributing_footprints=d.distributing_footprints*2),
    lambda d: replace(d, q4_pullback_policy="historical_corotational"),
    lambda d: replace(d, shell=replace(d.shell, s3_elements=((1, 2, 6),))),
    lambda d: replace(d, shell=replace(d.shell, j2_curve=object())),
    lambda d: replace(d, lines=(object(),)),
    lambda d: replace(d, footprints=(api.NodalRigidFootprint("rigid", (2, 3, 6), 102),)),
    lambda d: replace(d, shell_reference_policy=api.CURVED_Q4_POINT_REFERENCE_V1),
    lambda d: replace(d, footprint_chart_policy=api.GE_NODAL_FOOTPRINT_INCREMENT_ORIGIN_CHART_V2),
    lambda d: replace(d, modal_boundary_policy=api.GE_FREE_FREE_MODAL_ONLY_V1, fixed=()),
    lambda d: replace(d, bl5_material_policy=api.BL5_GENERALIZED_MATERIAL_V1),
    lambda d: replace(d, points=(api.PointConnection(3, 102),)),
    lambda d: replace(d, distributing_footprints=d.distributing_footprints+
        (replace(d.distributing_footprints[0], footprint_id="overlapping"),)),
))
def test_unsupported_combinations_duplicate_overlap_and_point_conflict_reject(prepared, change):
    with pytest.raises(ValueError): api.create_analysis(definition=change(prepared["definition"]))


def test_physical_fibre_admission_is_explicitly_excluded(prepared):
    from test_stiffened_shell_ge_physical_fibre_linear import _definition as fibre_definition
    definition = replace(prepared["definition"], beams=fibre_definition().beams)
    with pytest.raises(ValueError, match="physical-fibre"):
        api.create_analysis(definition=definition)


@pytest.fixture(scope="module")
def multi_public():
    definition = replace(_multi_definition(), points=(), distributing_footprints=(_multi_selection(),))
    owner = api.create_analysis(definition=definition)
    try:
        actions = (api.NodalAction("beam", 103, (0., 20., -10., 1., 0., 3.)),)
        first = owner.solve_linear(actions)
        yield definition, owner, actions, first
    finally:
        owner.close()


def test_public_multi_host_sparse_solve_dynamic_recovery_and_checkpoint_resume(multi_public):
    definition, owner, actions, first = multi_public
    compiled, start = owner._distributing_bindings[0]
    assert start == 0 and owner._C.shape == (6, 168) and sparse.isspmatrix_csr(owner._C)
    assert owner._C.has_canonical_format and owner.preflight()["reference_linear_factorizations"] == 1
    assert owner._C[:, owner._shell_count+6*len(owner._beam_ids):].nnz == 0
    record, = owner.recover(selection=api.GEReferenceDistributingFootprintSelection(("eight-host-area",)))
    assert record.shell_node_actions.shape == (15, 6) and len(record.host_element_ids) == 8
    assert record.configuration == "reference_linear" and not record.production_qualified
    np.testing.assert_array_equal(record.beam_action, -first.multipliers)
    np.testing.assert_allclose(record.shell_action[:3], first.multipliers[:3], atol=1e-12, rtol=0)
    np.testing.assert_allclose(record.shell_action[3:], first.multipliers[3:]+
                               np.cross(record.eccentricity_m, first.multipliers[:3]), atol=1e-12, rtol=0)
    virtual = np.sin(np.arange(owner._K.shape[0])+.2)
    local = compiled.gather(virtual).reshape(-1, 6)
    work = np.sum(record.shell_node_actions*local[:-1])+record.beam_action@local[-1]
    assert work == pytest.approx(-first.multipliers@(owner._C@virtual), abs=1e-10)
    np.testing.assert_array_equal(record.shell_node_actions[:, 5], np.zeros(15))
    checkpoint = owner.checkpoint()
    fresh = api.create_analysis(definition=definition)
    try:
        assert fresh._identity == owner._identity
        restored = fresh.restore(checkpoint)
        assert fresh.checkpoint() == checkpoint
        np.testing.assert_array_equal(_q(restored), _q(first))
        restored_record, = fresh.recover(selection=api.GEReferenceDistributingFootprintSelection(("eight-host-area",)))
        np.testing.assert_array_equal(restored_record.shell_node_actions, record.shell_node_actions)
        resumed = fresh.solve_linear(actions)
        assert resumed.generation == restored.generation+1
        assert fresh.preflight()["reference_linear_factorizations"] == 1
        np.testing.assert_array_equal(_q(resumed), _q(restored))
    finally:
        fresh.close()
    before = owner.checkpoint()
    with pytest.raises(ValueError, match="distributing"):
        owner.solve(actions)
    with pytest.raises(ValueError, match="distributing"):
        owner.modes(api.GEReferenceModesRequest())
    assert owner.checkpoint() == before and owner._finite_driver is None
