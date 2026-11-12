"""Bounded virgin B2 footprint descriptor checks, not mechanical qualification.

The dense oracle assembles/scatters separately and writes footprint equations
explicitly. It shares existing shell/beam stiffness and mass kernels and the
point law where present, so it is not an external formulation oracle.
"""
from dataclasses import asdict, FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import linalg, sparse
from scipy.sparse.linalg import ArpackNoConvergence

from anyfesolver import stiffened_shell as api
from anyfesolver.control import CancellationToken
from anyfesolver._b2_f10_f11_point_attachment import compile_reference_point
from anyfesolver._stiffened_shell_physical_q4_mass import physical_mass
import anyfesolver._stiffened_shell_b2_q4_panel_experimental as modal_owner
from test_stiffened_shell_b2_nodal_rigid_footprint import (
    _definition as _static_definition, _independent, _load, _owner,
)


def _definition(*, cells=2, points=False, multiple=False, selected_only=False):
    definition = _static_definition(cells=cells, points=points,
                                    selected_only=selected_only)
    mass = api.MassMetadata("centered-physical", (0., 0., 0.),
                           np.diag((.3, .3, .3, .003, .001, .002)))
    groups = definition.footprints
    if multiple:
        shell = definition.shells[-3]
        groups = (api.B2NodalRigidFootprint(
            "left-patch", shell.node_ids, 10+cells-3, (shell.node_ids,)),)+groups
    return replace(definition,
        shells=tuple(replace(s, density_kg_m3=2.) for s in definition.shells),
        members=tuple(replace(m, mass=mass) for m in definition.members),
        footprints=groups)


def _oracle(definition):
    k, footprint_c, shell_ids, beam_ids, _ = _independent(replace(definition, points=()))
    total = len(k)
    assert total < 200
    si = {node: i for i, node in enumerate(shell_ids)}
    bi = {node: len(si)+i for i, node in enumerate(beam_ids)}
    points = []
    mass = np.zeros_like(k)
    for shell in definition.shells:
        vertices = [definition.shell_nodes_m[n] for n in shell.node_ids]
        slots = [6*si[n]+j for n in shell.node_ids for j in range(6)]
        mass[np.ix_(slots, slots)] += physical_mass(vertices,
            density_kg_m3=shell.density_kg_m3, thickness_m=shell.thickness_m,
            reference_normal=shell.reference_normal)
    for member in definition.members:
        slots = [6*bi[n]+j for n in member.node_ids for j in range(6)]
        frame = member.local_dof_map()
        mass[np.ix_(slots, slots)] += frame.T@member.local_candidate_mass_matrix()@frame
    for point in definition.points:
        shell = next(s for s in definition.shells if s.shell_id == point.shell_id)
        member = next(m for m in definition.members if m.member_id == point.member_id)
        law = compile_reference_point(member,
            np.array([definition.shell_nodes_m[n] for n in shell.node_ids]),
            station=point.station, offset_local_m=point.offset_local_m)
        slots = ([6*si[n]+j for n in shell.node_ids for j in range(6)]+
                 [6*bi[n]+j for n in member.node_ids for j in range(6)])
        row = np.zeros((6, total)); row[:, slots] = law.constraint
        points.append(row)
    c = np.vstack(points+[footprint_c])
    t = definition.transformation.toarray()
    n = t.shape[1]; r = c.shape[0]
    # The physical shell drilling inertia is zero; descriptor B remains
    # singular and multipliers have exactly zero inertia, including footprints.
    a = np.block([[t.T@k@t, t.T@c.T], [c@t, np.zeros((r, r))]])
    b = linalg.block_diag(t.T@mass@t, np.zeros((r, r)))
    eigenvalues = linalg.eig(a, b, right=False)
    finite = eigenvalues[np.isfinite(eigenvalues)]
    assert np.max(abs(finite.imag), initial=0.) < 1e-8
    finite = np.sort(finite.real[finite.real > 0.])
    # A second bounded constraint reduction retains singular physical inertia.
    null = linalg.null_space(c@t)
    reduced = linalg.eig(null.T@t.T@k@t@null,
                        null.T@t.T@mass@t@null, right=False)
    reduced = np.sort(reduced.real[np.isfinite(reduced) & (reduced.real > 0.)])
    np.testing.assert_allclose(finite[:3], reduced[:3], rtol=2e-8)
    return np.sqrt(finite)/(2*np.pi), k, c, mass


@pytest.mark.parametrize("kwargs", ({}, {"selected_only": True},
    {"cells": 4, "multiple": True}, {"cells": 3, "points": True},
    {"cells": 5, "points": True, "multiple": True}))
def test_supported_virgin_modes_match_dense_physical_descriptor(request, kwargs):
    definition = _definition(**kwargs)
    expected, k, c, mass = _oracle(definition)
    owner = _owner(request, definition)
    np.testing.assert_allclose(owner._M.toarray(), mass, rtol=0., atol=0.)
    np.testing.assert_allclose(owner._C.toarray(), c, rtol=0., atol=0.)
    assert "modes" in owner.capabilities.operations
    assert not owner.capabilities.production_qualified
    assert "normal-spin locked" in owner.capabilities.spectral_domain
    assert "mechanical qualification incomplete" in owner.capabilities.spectral_domain
    result = owner.modes(api.B2ReferenceModesRequest(count=3))
    assert type(result) is api.B2FootprintReferenceModesResult
    assert isinstance(result, api.B2ReferenceModesResult)
    assert result.status == "ok", result.failure_reason
    assert result.footprint_ids == tuple(g.footprint_id for g in definition.footprints)
    assert result.footprint_law_id == owner.capabilities.footprint_law_id
    assert result.normal_spin == "locked" and result.mechanical_qualification == "incomplete"
    assert not result.production_qualified
    np.testing.assert_allclose(result.frequencies_hz, expected[:3], rtol=2e-8)
    vectors = np.array(result.physical_modes).T
    np.testing.assert_allclose(vectors.T@mass@vectors, np.eye(3), atol=1e-7)
    fixed = np.flatnonzero(np.diff(definition.transformation.indptr) == 0)
    np.testing.assert_array_equal(vectors[fixed], 0.)
    for frequency, vector, multiplier in zip(result.frequencies_hz,
            result.physical_modes, result.constraint_multiplier_modes):
        vector = np.asarray(vector); multiplier = np.asarray(multiplier)
        residual = k@vector+c.T@multiplier-(2*np.pi*frequency)**2*mass@vector
        bound = abs(k)@abs(vector)+abs(c.T)@abs(multiplier)+(2*np.pi*frequency)**2*abs(mass)@abs(vector)
        assert np.linalg.norm(definition.transformation.T@residual) <= 1e-8*np.linalg.norm(bound)
        assert np.linalg.norm(c@vector) <= 1e-9*max(np.linalg.norm(abs(c)@abs(vector)), 1.)
    assert max(result.equation_residuals) <= 1e-8
    assert max(result.constraint_residuals) <= 1e-9
    assert result.factorization_count == 1 and result.solve_count > 0
    components = ("ux", "uy", "uz", "rx", "ry", "rz")
    expected_rows = [(p.point_id, None, None, None, component)
                     for p in definition.points for component in components]
    expected_rows += [(None, g.footprint_id, node, g.beam_node, component)
                     for g in definition.footprints for node in g.shell_nodes
                     for component in components]
    assert [(r.point_id, r.footprint_id, r.shell_node_id, r.beam_node_id, r.component)
            for r in result.constraint_rows] == expected_rows
    assert [r.row_index for r in result.constraint_rows] == list(range(len(c)))
    for row in result.constraint_rows[6*len(definition.points):]:
        assert row.coordinate_frame == "global"
        np.testing.assert_array_equal(row.reference_frame_columns, np.eye(3))
    assert all(len(m) == len(c) for m in result.constraint_multiplier_modes)
    with pytest.raises(FrozenInstanceError):
        result.constraint_rows[0].component = "changed"
    if definition.points:
        assert result.point_multiplier_modes == tuple(m[:6*len(definition.points)]
                                                      for m in result.constraint_multiplier_modes)
    else:
        assert result.point_multiplier_modes == ()


def test_origin_and_affine_coordinate_rescaling_invariance(request):
    definition = _definition(cells=4, multiple=True)
    reference = _owner(request, definition).modes(api.B2ReferenceModesRequest())
    origin = np.array((17., -8., 3.))
    changed = replace(definition,
        shell_nodes_m={n: tuple(np.asarray(p)+origin) for n, p in definition.shell_nodes_m.items()},
        members=tuple(replace(m, reference_points_m=tuple(tuple(np.asarray(p)+origin)
                        for p in m.reference_points_m)) for m in definition.members),
        transformation=definition.transformation@sparse.diags(
            np.geomspace(1e-3, 1e3, definition.transformation.shape[1])))
    result = _owner(request, changed).modes(api.B2ReferenceModesRequest(shift_hz=.2))
    assert reference.status == result.status == "ok", result.failure_reason
    np.testing.assert_allclose(result.frequencies_hz, reference.frequencies_hz, rtol=2e-8)


def test_rotated_mixed_points_retain_actual_material_gap_frames(request):
    definition = _definition(cells=3, points=True)
    axis = np.array((1., 2., -1.)); axis /= np.linalg.norm(axis)
    cross = np.array(((0., -axis[2], axis[1]), (axis[2], 0., -axis[0]),
                      (-axis[1], axis[0], 0.)))
    rotation = np.eye(3)+np.sin(.63)*cross+(1-np.cos(.63))*(cross@cross)
    changed = replace(definition,
        shell_nodes_m={n: tuple(rotation@p) for n, p in definition.shell_nodes_m.items()},
        shells=tuple(replace(s, reference_normal=tuple(rotation@s.reference_normal))
                     for s in definition.shells),
        members=tuple(replace(m, local_y_global=tuple(rotation@m.local_y_global),
            reference_points_m=tuple(tuple(rotation@p) for p in m.reference_points_m))
            for m in definition.members))
    base = _owner(request, definition).modes(api.B2ReferenceModesRequest())
    owner = _owner(request, changed)
    result = owner.modes(api.B2ReferenceModesRequest())
    expected, k, c, mass = _oracle(changed)
    assert base.status == result.status == "ok", result.failure_reason
    np.testing.assert_allclose(result.frequencies_hz, base.frequencies_hz, rtol=2e-8)
    np.testing.assert_allclose(result.frequencies_hz, expected[:3], rtol=2e-8)
    point = changed.points[0]
    shell = next(s for s in changed.shells if s.shell_id == point.shell_id)
    member = next(m for m in changed.members if m.member_id == point.member_id)
    law = compile_reference_point(member,
        np.array([changed.shell_nodes_m[n] for n in shell.node_ids]), station=point.station)
    assert not np.allclose(law.reference_frames[0], np.eye(3))
    for j, row in enumerate(result.constraint_rows[:6]):
        assert row.coordinate_frame == ("point_master_shell_reference" if j < 3
                                        else "point_beam_reference")
        np.testing.assert_array_equal(row.reference_frame_columns, law.reference_frames[0 if j < 3 else 1])
    # Work is recovered using the actual C transpose, never relabeled point
    # multipliers as global forces/couples. Footprint rows remain global.
    vector = np.asarray(result.physical_modes[0])
    multiplier = np.asarray(result.constraint_multiplier_modes[0])
    virtual = np.linspace(-.03, .07, len(vector))
    assert abs(virtual@(c.T@multiplier)-multiplier@(c@virtual)) < 1e-10
    residual = k@vector+c.T@multiplier-(2*np.pi*result.frequencies_hz[0])**2*mass@vector
    assert np.linalg.norm(changed.transformation.T@residual) < 1e-7


@pytest.mark.parametrize("missing", ("shell", "beam"))
def test_missing_physical_mass_keeps_static_only_capability(request, missing):
    definition = _definition()
    if missing == "shell":
        definition = replace(definition, shells=(replace(definition.shells[0], density_kg_m3=None),)+definition.shells[1:])
    else:
        definition = replace(definition, members=(replace(definition.members[0], mass=None),)+definition.members[1:])
    owner = _owner(request, definition)
    assert "modes" not in owner.capabilities.operations
    with pytest.raises(ValueError, match="explicit shell density and beam physical mass"):
        owner.modes(api.B2ReferenceModesRequest())
    assert owner.solve_linear(_load(definition)).status == "completed"


@pytest.mark.parametrize("kind", ("MPC", "offset"))
def test_nonvirgin_or_mpc_plan_is_explicitly_rejected(request, kind):
    definition = _definition()
    if kind == "MPC":
        t = definition.transformation.tolil()
        row = t[:, 0].nonzero()[0][0]; t[row, 1] = .2
        definition = replace(definition, transformation=t.tocsr())
    else:
        offset = definition.prescribed_offset.copy(); offset[0] = 1e-5
        definition = replace(definition, prescribed_offset=offset)
    owner = _owner(request, definition)
    assert "modes" not in owner.capabilities.operations
    with pytest.raises(ValueError, match="without MPCs" if kind == "MPC" else "zero prescribed offset"):
        owner.modes(api.B2ReferenceModesRequest())
    assert owner.solve_linear(_load(definition)).status == "completed"


@pytest.mark.parametrize("stage", ("before", "iteration", "after"))
def test_cancellation_discards_pairs_and_keeps_static_factor(request, monkeypatch, stage):
    definition = _definition(cells=3, points=True)
    owner = _owner(request, definition)
    token = CancellationToken()
    eigs = modal_owner.eigs
    def interrupted(operator, **kwargs):
        if stage == "iteration":
            token.cancel("during modal work")
            operator@np.ones(operator.shape[0])
            raise AssertionError("must cancel")
        result = eigs(operator, **kwargs)
        token.cancel("after modal work")
        return result
    if stage == "before":
        token.cancel("before modal work")
    else:
        monkeypatch.setattr(modal_owner, "eigs", interrupted)
    handle = owner._handle
    before = asdict(owner.solve_linear(_load(definition)))
    count = handle.solve_count
    result = owner.modes(api.B2ReferenceModesRequest(), cancellation=token)
    assert result.status == "cancelled"
    _no_pairs(result)
    assert result.factorization_count == (0 if stage == "before" else 1)
    assert owner._handle is handle and handle.solve_count == count
    after = asdict(owner.solve_linear(_load(definition)))
    before.pop("solve_count"); after.pop("solve_count")
    assert before == after


@pytest.mark.parametrize("fixed", ((), (0, 1, 2, 3, 4)))
def test_free_or_one_missing_rigid_restraint_rejects_before_modal_work(monkeypatch, fixed):
    definition = _definition()
    total = definition.transformation.shape[0]
    free = [i for i in range(total) if i not in fixed]
    definition = replace(definition, transformation=sparse.eye(total, format="csr")[:, free])
    factor_calls = []
    # A rounded static factor success must not grant free/partial support modes.
    # Exercise the admission even when that backend admits the singular KKT.
    def nominal_factor(*args, **kwargs):
        factor_calls.append(args)
        return SimpleNamespace(status="ok", factorization_count=1, solve_count=0)
    monkeypatch.setattr(modal_owner, "factorize_cached", nominal_factor)
    monkeypatch.setattr(modal_owner, "eigs", lambda *args, **kwargs:
                        pytest.fail("under-supported spectrum reached eigensolve"))
    owner = api.create_analysis(definition=definition)
    assert len(factor_calls) == 1
    assert "modes" not in owner.capabilities.operations
    with pytest.raises(ValueError, match="all six physical rigid fields"):
        owner.modes(api.B2ReferenceModesRequest())
    assert len(factor_calls) == 1


def test_support_detectability_uses_actual_physical_rows_without_global_dense_audit(monkeypatch):
    definition = _definition()
    # Six numerical support rows need not detect six rigid motions: these
    # translations all lie on the bottom x-axis, leaving rotation about x free.
    total = definition.transformation.shape[0]
    fixed = (0, 1, 2, 6, 7, 8)
    free = [i for i in range(total) if i not in fixed]
    t = sparse.eye(total, format="csr")[:, free]
    beam_positions = {node: point for member in definition.members
                      for node, point in zip(member.node_ids, member.reference_points_m)}
    original_qr = np.linalg.qr
    original_svd = np.linalg.svd
    def bounded_qr(matrix, **kwargs):
        assert matrix.shape[0] <= 12 and matrix.shape[1] == 6
        return original_qr(matrix, **kwargs)
    def bounded_svd(matrix, **kwargs):
        assert matrix.shape == (6, 6)
        return original_svd(matrix, **kwargs)
    monkeypatch.setattr(np.linalg, "qr", bounded_qr)
    monkeypatch.setattr(np.linalg, "svd", bounded_svd)
    check = modal_owner._supports_all_panel_rigid_fields
    assert not check(t, definition.shell_nodes_m, beam_positions)
    assert check(definition.transformation, definition.shell_nodes_m, beam_positions)


def _no_pairs(result):
    assert result.frequencies_hz == result.physical_modes == ()
    assert result.point_multiplier_modes == result.constraint_multiplier_modes == ()
    assert result.equation_residuals == result.constraint_residuals == ()
    assert result.constraint_rows  # Provenance is meaningful even on failure.


@pytest.mark.parametrize("failure", ("arpack", "backend", "nonfinite"))
def test_failure_never_publishes_partial_modes_or_changes_static(request, monkeypatch, failure):
    definition = _definition()
    owner = _owner(request, definition)
    handle = owner._handle
    before = asdict(owner.solve_linear(_load(definition))); count = handle.solve_count
    if failure == "arpack":
        def bad(*args, **kwargs):
            raise ArpackNoConvergence("controlled partial spectrum", np.array((1.,)),
                                       np.ones((1, 1)))
        monkeypatch.setattr(modal_owner, "eigs", bad)
    elif failure == "nonfinite":
        monkeypatch.setattr(modal_owner, "eigs", lambda operator, **kwargs:
            (np.full(3, np.nan), np.ones((operator.shape[0], 3))))
    else:
        def bad_solve(rhs):
            raise RuntimeError("Sparse solve produced NaN/Inf values")
        failed = SimpleNamespace(status="ok", factorization_count=1, solve_count=1, solve=bad_solve)
        monkeypatch.setattr(modal_owner, "factorize_cached", lambda *args, **kwargs: failed)
    result = owner.modes(api.B2ReferenceModesRequest())
    assert result.status == "failed" and result.failure_reason
    _no_pairs(result)
    assert result.factorization_count == 1
    assert owner._handle is handle and handle.solve_count == count
    after = asdict(owner.solve_linear(_load(definition)))
    before.pop("solve_count"); after.pop("solve_count")
    assert before == after


def test_empty_footprint_preserves_legacy_point_modal_result(request):
    from test_stiffened_shell_b2_reference_modes import _definition as legacy
    definition, _ = legacy()
    result = _owner(request, definition).modes(api.B2ReferenceModesRequest())
    assert result.status == "ok", result.failure_reason
    assert type(result) is api.B2ReferenceModesResult
    assert "footprint_ids" not in asdict(result)
