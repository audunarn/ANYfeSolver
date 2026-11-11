"""Focused curved shell skin geometry/work checks on the delivered case.

One session-scoped fixture performs the single existing panel gravity
preparation (``prepare_case_models``) and builds both face-local curved skin
maps from the actual delivered sphere and panel models.  The first tests are
the slice-7 discriminators: exact 320 S3 + 100 Q4 face inventory, zero-pose
agreement with the authoritative per-element skin corners, and shared-node
work accumulation onto the authoritative structural DOFs.  The remaining
checks reuse the finite-skin regression stencils (director error 1e-13,
force pullback 1e-14, virtual work 1e-13, derivative FD 1e-5) with
deterministic normalized unit-probe fields, and independently compute
positions, resultants and rotations rather than duplicating the map code.
The geometry service is development-only; no contact solve is admitted here.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from anyfesolver import BeamElement
from anyfesolver.corotational import (
    _rotation_right_jacobian,
    rotation_matrix_from_vector,
)
from anyfesolver.general_contact.curved_shell_skin import (
    CurvedShellSkinError,
    CurvedShellSkinMap,
    TOTAL_ROTATION_BOUND_RAD,
)
from anyfesolver.mesh_gen import InterpolatedBeamShellMPCElement

from examples.elastic_sphere_stiffened_panel import (
    PLATE_THICKNESS,
    PLATE_TOP_Z,
    prepare_case_models,
)
from examples.elastic_sphere_panel_contact_skins import build_skin_maps

DIRECTOR_TOLERANCE = 1.0e-13
FORCE_PULLBACK_TOLERANCE = 1.0e-14
VIRTUAL_WORK_TOLERANCE = 1.0e-13
DERIVATIVE_RELATIVE_TOLERANCE = 1.0e-5


def _plate_shell_ids(panel_model):
    return sorted(
        element_id for element_id, element in panel_model.mesh.elements.items()
        if not isinstance(element, (BeamElement, InterpolatedBeamShellMPCElement))
    )


def _independent_reference_corners(model, face):
    """Zero-pose skin corners from declared case data, not the map code."""
    element = model.mesh.elements[face.element_id]
    coordinates = np.asarray(element.get_node_coordinates(model.mesh), dtype=float)
    if len(face.node_ids) == 3:
        director = np.asarray(element.reference_normal, dtype=float)
    else:
        director = np.cross(coordinates[1] - coordinates[0],
                            coordinates[3] - coordinates[0])
        director = director / np.linalg.norm(director)
    offset = 0.5 * float(element.thickness)
    return coordinates + offset * director, offset, director


def _independent_positions(model, skin_map, q):
    """Face-local corners from the libm exponential, not the Taylor map."""
    expected = np.empty_like(skin_map.reference)
    for index, node in enumerate(skin_map.corner_nodes):
        dofs = model.mesh.dof_manager.get_node_dofs(int(node))
        nodal = q[np.asarray(dofs[:6], dtype=np.intp)]
        rotated = rotation_matrix_from_vector(nodal[3:]) @ skin_map.directors[index]
        expected[index] = (skin_map.reference[index] + nodal[:3]
                           + skin_map.offsets[index] * rotated)
    return expected


def _uniform_rotation(model, skin_map, theta):
    q = np.zeros(skin_map.n_dof)
    for node in {int(n) for n in skin_map.corner_nodes}:
        dofs = model.mesh.dof_manager.get_node_dofs(node)
        q[np.asarray(dofs[3:6], dtype=np.intp)] = theta
    return q


def _uniform_translation(model, skin_map, shift):
    q = np.zeros(skin_map.n_dof)
    for node in {int(n) for n in skin_map.corner_nodes}:
        dofs = model.mesh.dof_manager.get_node_dofs(node)
        q[np.asarray(dofs[:3], dtype=np.intp)] = shift
    return q


def _local_probe_field(model, skin_map, seed):
    """Deterministic small nodal translation/rotation probe."""
    rng = np.random.default_rng(seed)
    q = np.zeros(skin_map.n_dof)
    for node in sorted({int(n) for n in skin_map.corner_nodes}):
        dofs = model.mesh.dof_manager.get_node_dofs(node)
        dofs = np.asarray(dofs[:6], dtype=np.intp)
        q[dofs[:3]] = 1.0e-4 * rng.normal(size=3)
        q[dofs[3:]] = 1.0e-3 * rng.normal(size=3)
    return q


@pytest.fixture(scope="session")
def prepared():
    """The one existing gravity/model preparation (single panel gravity solve)."""
    return prepare_case_models()


@pytest.fixture(scope="session")
def sphere(prepared):
    panel_model, sphere_model, _geometry, _load, _result, _q = prepared
    sphere_map, _panel_map = build_skin_maps(panel_model, sphere_model)
    return sphere_model, sphere_map


@pytest.fixture(scope="session")
def panel(prepared):
    panel_model, _sphere_model, _geometry, _load, _result, displacements = prepared
    _sphere_map, panel_map = build_skin_maps(panel_model, _sphere_model)
    return panel_model, panel_map, displacements


def test_batched_centre_normals_match_scalar_and_bound_validation(sphere, panel, monkeypatch):
    for model, skin, q in ((sphere[0], sphere[1], np.zeros(sphere[1].n_dof)), panel):
        pose = skin.pose(q)
        expected = np.asarray([skin.face_normal(pose, i,
            np.full(3, 1./3.) if len(face.node_ids)==3 else np.zeros(2))
            for i, face in enumerate(skin.faces)])
        calls = []
        original = skin._require_bound_geometry
        with monkeypatch.context() as patch:
            def check_bound(model):
                calls.append(True)
                return original(model)
            patch.setattr(skin, '_require_bound_geometry', check_bound)
            actual = skin.face_centre_normals(pose)
        np.testing.assert_array_equal(actual, expected)
        assert len(calls)==2  # Fixed admission/exit cost, not one global scan per face.


def test_batched_normals_refuse_stale_foreign_and_mutating_geometry(sphere, panel, monkeypatch):
    model, skin = sphere
    pose = skin.pose(np.zeros(skin.n_dof))
    with pytest.raises(CurvedShellSkinError):
        panel[1].face_centre_normals(pose)
    node = next(iter(model.mesh.nodes.values()))
    original_x = node.x
    try:
        node.x += .01
        with pytest.raises(CurvedShellSkinError, match='changed'):
            skin.face_centre_normals(pose)
    finally:
        node.x = original_x
    original = skin._face_normal
    try:
        with monkeypatch.context() as patch:
            def mutate_after_normal(*args):
                result = original(*args)
                node.x = original_x + .01
                return result
            patch.setattr(skin, '_face_normal', mutate_after_normal)
            with pytest.raises(CurvedShellSkinError, match='changed'):
                skin.face_centre_normals(pose)
    finally:
        node.x = original_x
    for invalid in (np.nan, 0.):
        saved = pose.positions.copy()
        try:
            pose.positions[:3] = invalid
            with pytest.raises(CurvedShellSkinError, match='degenerate normal'):
                skin.face_centre_normals(pose)
        finally:
            pose.positions[:] = saved


def test_actual_model_inventory_and_zero_pose_reference(sphere, panel):
    """Discriminator: exact 320 S3 + 100 Q4 faces and authoritative corners."""
    sphere_model, sphere_map = sphere
    panel_model, panel_map, _q = panel
    assert len(sphere_model.mesh.elements) == 320
    assert all(len(e.node_ids) == 3 for e in sphere_model.mesh.elements.values())
    assert len(_plate_shell_ids(panel_model)) == 100
    assert len(sphere_map.faces) == 320 and len(sphere_map.corner_keys) == 960
    assert len(panel_map.faces) == 100 and len(panel_map.corner_keys) == 400
    assert len(set(sphere_map.corner_keys)) == 960
    assert len(set(panel_map.corner_keys)) == 400
    for skin_map, model in ((sphere_map, sphere_model), (panel_map, panel_model)):
        zero = np.zeros(skin_map.n_dof)
        pose = skin_map.pose(zero)
        for face, (start, end) in zip(skin_map.faces, skin_map.face_slices):
            expected, _offset, _director = _independent_reference_corners(model, face)
            np.testing.assert_allclose(
                pose.positions[start:end], expected, rtol=0, atol=1.0e-15)


def test_gravity_preloaded_panel_skin_recovery(panel):
    """Discriminator: panel skin recovered at the accepted gravity state."""
    panel_model, panel_map, displacements = panel
    pose = panel_map.pose(displacements)
    expected = _independent_positions(panel_model, panel_map, displacements)
    np.testing.assert_allclose(
        pose.positions, expected, rtol=0, atol=DIRECTOR_TOLERANCE)
    centre = min(panel_model.mesh.nodes.values(),
                 key=lambda node: abs(node.x) + abs(node.y))
    centre_top_z = PLATE_TOP_Z + float(displacements[centre.dofs[2]])
    corners = [index for index, node in enumerate(panel_map.corner_nodes)
               if int(node) == int(centre.id)]
    assert corners
    np.testing.assert_allclose(
        pose.positions[corners, 2], centre_top_z, rtol=0, atol=1.0e-15)


def test_shared_node_work_accumulation(sphere, panel):
    """Discriminator: shared-node forces/moments accumulate on authoritative DOFs."""
    sphere_model, sphere_map = sphere
    panel_model, panel_map, _q = panel
    for model, skin_map in ((sphere_model, sphere_map), (panel_model, panel_map)):
        pose = skin_map.pose(np.zeros(skin_map.n_dof))
        counts: dict[int, int] = {}
        for node in skin_map.corner_nodes:
            counts[int(node)] = counts.get(int(node), 0) + 1
        node = max(counts, key=counts.get)
        corners = [i for i, n in enumerate(skin_map.corner_nodes) if int(n) == node]
        forces = np.zeros((len(skin_map.corner_keys), 3))
        for k, index in enumerate(corners):
            forces[index] = np.eye(3)[k % 3]
        pulled = skin_map.spin_pullback(pose, forces)
        independent = np.zeros(skin_map.n_dof)
        dofs = np.asarray(model.mesh.dof_manager.get_node_dofs(node)[:6],
                          dtype=np.intp)
        for index in corners:
            independent[dofs[:3]] += forces[index]
            independent[dofs[3:]] += skin_map.offsets[index] * np.cross(
                skin_map.directors[index], forces[index])
        np.testing.assert_allclose(
            pulled, independent, rtol=0, atol=FORCE_PULLBACK_TOLERANCE)
        assert np.count_nonzero(pulled) <= 6


def test_shared_sphere_corner_distinctness(sphere, panel):
    """Shared sphere nodes keep distinct per-element corners; panel corners agree."""
    sphere_model, sphere_map = sphere
    panel_model, panel_map, _q = panel
    theta = 0.05 * np.array((1.0, 1.0, 1.0)) / math.sqrt(3.0)
    for model, skin_map, expect_distinct in (
            (sphere_model, sphere_map, True), (panel_model, panel_map, False)):
        pose = skin_map.pose(_uniform_rotation(model, skin_map, theta))
        counts: dict[int, int] = {}
        for node in skin_map.corner_nodes:
            counts[int(node)] = counts.get(int(node), 0) + 1
        node = max(counts, key=counts.get)
        corners = [i for i, n in enumerate(skin_map.corner_nodes) if int(n) == node]
        points = pose.positions[corners]
        spread = max(
            float(np.linalg.norm(points[i] - points[j]))
            for i in range(len(corners)) for j in range(i + 1, len(corners)))
        distinct_directors = {
            tuple(skin_map.directors[i]) for i in corners}
        if expect_distinct:
            assert len(distinct_directors) >= 2
            assert spread > 1.0e-4
        else:
            assert spread <= 1.0e-15


def test_rigid_translation_of_each_body(sphere, panel):
    """Rigid translations shift each body's skin exactly."""
    shift = np.array((0.03, -0.02, 0.05))
    for model, skin_map in sphere, panel[:2]:
        zero = skin_map.pose(np.zeros(skin_map.n_dof))
        moved = skin_map.pose(_uniform_translation(model, skin_map, shift))
        np.testing.assert_allclose(
            moved.positions, zero.positions + shift, rtol=0, atol=1.0e-15)


def test_rigid_rotation_virtual_work_of_each_body(sphere, panel):
    """Rigid rotations match the independent exponential and both work maps."""
    theta = 0.4 * np.array((1.0, 2.0, 3.0)) / math.sqrt(14.0)
    for model, skin_map in (sphere, panel[:2]):
        q = _uniform_rotation(model, skin_map, theta)
        pose = skin_map.pose(q)
        expected = _independent_positions(model, skin_map, q)
        np.testing.assert_allclose(
            pose.positions, expected, rtol=0, atol=DIRECTOR_TOLERANCE)
        rng = np.random.default_rng(71)
        forces = rng.normal(size=pose.positions.shape)
        forces /= np.linalg.norm(forces)
        pulled = skin_map.spin_pullback(pose, forces)
        independent = np.zeros(skin_map.n_dof)
        delta = rng.normal(size=skin_map.n_dof)
        delta /= np.linalg.norm(delta)
        virtual = delta.copy()
        for index, dofs in enumerate(skin_map.dofs):
            rotated = rotation_matrix_from_vector(q[dofs[3:]]) @ skin_map.directors[index]
            independent[dofs[:3]] += forces[index]
            independent[dofs[3:]] += skin_map.offsets[index] * np.cross(
                rotated, forces[index])
        for node in {int(n) for n in skin_map.corner_nodes}:
            dofs = np.asarray(model.mesh.dof_manager.get_node_dofs(node)[:6],
                              dtype=np.intp)
            virtual[dofs[3:]] = _rotation_right_jacobian(-q[dofs[3:]]) @ delta[dofs[3:]]
        np.testing.assert_allclose(
            pulled, independent, rtol=0, atol=FORCE_PULLBACK_TOLERANCE)
        work_error = abs(float(
            pulled @ virtual - forces.ravel() @ (pose.coordinate_map @ delta)))
        assert work_error <= VIRTUAL_WORK_TOLERANCE


def test_local_deformation_probe_of_each_body(sphere, panel):
    """Distinct nodal rotations map through the same exponential field."""
    for index, (model, skin_map) in enumerate((sphere, panel[:2])):
        q = _local_probe_field(model, skin_map, seed=7 + index)
        pose = skin_map.pose(q)
        expected = _independent_positions(model, skin_map, q)
        np.testing.assert_allclose(
            pose.positions, expected, rtol=0, atol=DIRECTOR_TOLERANCE)


def test_fixed_spatial_force_pullback_derivative(sphere, panel):
    """FD derivative of the fixed-force spin pullback matches the map."""
    for index, (model, skin_map) in enumerate((sphere, panel[:2])):
        q = _local_probe_field(model, skin_map, seed=21 + index)
        rng = np.random.default_rng(31 + index)
        forces = rng.normal(size=(len(skin_map.corner_keys), 3))
        forces /= np.linalg.norm(forces)
        direction = rng.normal(size=skin_map.n_dof)
        direction /= np.linalg.norm(direction)
        derivative = skin_map.spin_force_derivative(skin_map.pose(q), forces)

        def pullback(at):
            return skin_map.spin_pullback(skin_map.pose(at), forces)

        base = derivative @ direction
        for step in (1.0e-6, 3.0e-7):
            finite = (pullback(q + step * direction)
                      - pullback(q - step * direction)) / (2.0 * step)
            relative = (np.linalg.norm(finite - base)
                        / max(np.linalg.norm(base), 1.0e-12))
            assert relative <= DERIVATIVE_RELATIVE_TOLERANCE


def test_physical_offsets_and_authoritative_dofs(sphere, panel):
    """Offsets are the declared physical tops; pullback lands on node DOFs."""
    sphere_model, sphere_map = sphere
    panel_model, panel_map, _q = panel
    zero = sphere_map.pose(np.zeros(sphere_map.n_dof))
    radii = np.linalg.norm(zero.positions, axis=1)
    assert float(np.max(radii)) <= 0.5 + 1.0e-12
    assert float(np.max(radii)) < 0.5
    assert float(np.min(radii)) > 0.49
    panel_zero = panel_map.pose(np.zeros(panel_map.n_dof))
    for face, (start, end) in zip(panel_map.faces, panel_map.face_slices):
        coordinates = np.asarray(
            panel_model.mesh.elements[face.element_id].get_node_coordinates(
                panel_model.mesh), dtype=float)
        np.testing.assert_allclose(
            panel_zero.positions[start:end, 2],
            coordinates[:, 2] + 0.5 * PLATE_THICKNESS, rtol=0, atol=1.0e-15)
    for model, skin_map in (sphere, panel[:2]):
        pose = skin_map.pose(np.zeros(skin_map.n_dof))
        corner = len(skin_map.corner_keys) // 2
        forces = np.zeros((len(skin_map.corner_keys), 3))
        forces[corner] = (0.0, 0.0, 1.0)
        pulled = skin_map.spin_pullback(pose, forces)
        dofs = skin_map.dofs[corner]
        node_dofs = np.asarray(
            model.mesh.dof_manager.get_node_dofs(
                int(skin_map.corner_nodes[corner]))[:6], dtype=np.intp)
        np.testing.assert_allclose(
            pulled[node_dofs[:3]], forces[corner], rtol=0, atol=0.0)
        expected_moment = skin_map.offsets[corner] * np.cross(
            skin_map.directors[corner], forces[corner])
        np.testing.assert_allclose(
            pulled[node_dofs[3:]], expected_moment, rtol=0,
            atol=FORCE_PULLBACK_TOLERANCE)
        mask = np.ones(skin_map.n_dof, dtype=bool)
        mask[node_dofs] = False
        assert not np.any(pulled[mask])
        assert np.array_equal(dofs, node_dofs)


def test_refusals_and_nonmutation(sphere, panel):
    """Invalid poses, bounds, mutations and declarations refuse; nothing mutates."""
    sphere_model, sphere_map = sphere
    panel_model, panel_map, _q = panel
    n_sphere = sphere_map.n_dof
    good = np.zeros(n_sphere)
    before = np.asarray(sphere_model.mesh.get_node(
        int(sphere_map.corner_nodes[0])).coords(), dtype=float).copy()
    sphere_map.pose(good)
    assert np.array_equal(good, np.zeros(n_sphere))
    with pytest.raises(CurvedShellSkinError):
        sphere_map.pose(good[:-1])
    nan = good.copy(); nan[0] = np.nan
    with pytest.raises(CurvedShellSkinError):
        sphere_map.pose(nan)
    with pytest.raises(CurvedShellSkinError):
        sphere_map.pose(good.astype(bool))
    with pytest.raises(CurvedShellSkinError):
        sphere_map.pose([0.0] * n_sphere)
    over = _uniform_rotation(
        sphere_model, sphere_map,
        np.array((TOTAL_ROTATION_BOUND_RAD + 0.1, 0.0, 0.0)))
    with pytest.raises(CurvedShellSkinError) as excinfo:
        sphere_map.pose(over)
    assert excinfo.value.code == "rotation_bound_exceeded"
    inverted = np.zeros(panel_map.n_dof)
    face = panel_map.faces[0]
    for node in face.node_ids[1:3]:
        dofs = panel_model.mesh.dof_manager.get_node_dofs(int(node))
        inverted[dofs[0]] = -2.0
    with pytest.raises(CurvedShellSkinError) as excinfo:
        panel_map.pose(inverted)
    assert excinfo.value.code == "invalid_geometry"
    node = int(sphere_map.corner_nodes[0])
    x, y, z = before
    sphere_model.mesh.set_node_coordinates(node, x + 1.0e-3, y, z)
    with pytest.raises(CurvedShellSkinError) as excinfo:
        sphere_map.pose(good)
    assert excinfo.value.code == "mutated_bound_geometry"
    sphere_model.mesh.set_node_coordinates(node, x, y, z)
    sphere_map.pose(good)
    after = np.asarray(sphere_model.mesh.get_node(node).coords(), dtype=float)
    assert np.array_equal(before, after)
    with pytest.raises(CurvedShellSkinError):
        CurvedShellSkinMap(sphere_model, ())
    with pytest.raises(CurvedShellSkinError):
        CurvedShellSkinMap(sphere_model, ((1, 1, "top"), (1, 1, "top")))
    with pytest.raises(CurvedShellSkinError):
        CurvedShellSkinMap(sphere_model, ((1, 9999, "top"),))
    beam_id = next(element_id for element_id, element
                   in panel_model.mesh.elements.items()
                   if isinstance(element, BeamElement))
    with pytest.raises(CurvedShellSkinError):
        CurvedShellSkinMap(panel_model, ((2, beam_id, "top"),))


def test_wrong_map_pose_and_shape_refusals(sphere, panel):
    """Poses belong to their own map; displacement/force shapes are exact."""
    _sphere_model, sphere_map = sphere
    _panel_model, panel_map, _q = panel
    sphere_pose = sphere_map.pose(np.zeros(sphere_map.n_dof))
    panel_zero = np.zeros((len(panel_map.dofs), 3))
    for consumer in (panel_map.spin_pullback, panel_map.coordinate_pullback,
                     panel_map.spin_force_derivative):
        with pytest.raises(CurvedShellSkinError) as excinfo:
            consumer(sphere_pose, panel_zero)
        assert excinfo.value.code == "invalid_request"
    with pytest.raises(CurvedShellSkinError) as excinfo:
        panel_map.face_point(sphere_pose, 0, (0.0, 0.0))
    assert excinfo.value.code == "invalid_request"
    with pytest.raises(CurvedShellSkinError) as excinfo:
        panel_map.face_normal(sphere_pose, 0, (0.0, 0.0))
    assert excinfo.value.code == "invalid_request"
    with pytest.raises(CurvedShellSkinError) as excinfo:
        panel_map.face_force_pullback(sphere_pose, 0, (0.0, 0.0), (0.0, 0.0, 1.0))
    assert excinfo.value.code == "invalid_request"
    # A two-dimensional displacement with the correct total size refuses.
    flat = np.zeros(sphere_map.n_dof)
    for reshaped in (flat.reshape(1, -1), flat.reshape(-1, 1)):
        with pytest.raises(CurvedShellSkinError) as excinfo:
            sphere_map.pose(reshaped)
        assert excinfo.value.code == "invalid_displacement"
    # The derivative input must be exactly (n_corners, 3), not a flat vector.
    with pytest.raises(CurvedShellSkinError):
        sphere_map.spin_force_derivative(
            sphere_pose, np.zeros(3 * len(sphere_map.dofs)))
    # Same-map use still works after the refusals.
    pulled = sphere_map.spin_pullback(
        sphere_pose, np.zeros((len(sphere_map.dofs), 3)))
    assert pulled.shape == (sphere_map.n_dof,)
    assert not pulled.any()


def test_bound_identity_refuses_thickness_and_director_changes(sphere, panel):
    """Changed element thickness/director refuse as bound-geometry mutations."""
    sphere_model, sphere_map = sphere
    panel_model, panel_map, _q = panel
    zero_sphere = np.zeros(sphere_map.n_dof)
    zero_panel = np.zeros(panel_map.n_dof)
    # A changed Q4 thickness changes the physical offset: refuse, restore.
    face = panel_map.faces[0]
    element = panel_model.mesh.elements[face.element_id]
    thickness = float(element.thickness)
    element.thickness = 1.01 * thickness
    try:
        with pytest.raises(CurvedShellSkinError) as excinfo:
            panel_map.pose(zero_panel)
        assert excinfo.value.code == "mutated_bound_geometry"
    finally:
        element.thickness = thickness
    panel_map.pose(zero_panel)
    # A pose saved before an offset mutation refuses in every consumer
    # until the bound geometry is restored (stale saved pose, not only
    # pose creation).
    saved = panel_map.pose(zero_panel)
    forces = np.zeros((len(panel_map.dofs), 3))
    element.thickness = 1.01 * thickness
    try:
        with pytest.raises(CurvedShellSkinError) as excinfo:
            panel_map.spin_pullback(saved, forces)
        assert excinfo.value.code == "mutated_bound_geometry"
        with pytest.raises(CurvedShellSkinError) as excinfo:
            panel_map.face_point(saved, 0, (0.0, 0.0))
        assert excinfo.value.code == "mutated_bound_geometry"
    finally:
        element.thickness = thickness
    assert panel_map.spin_pullback(saved, forces).shape == (panel_map.n_dof,)
    # The same stale-pose refusal after a bound node coordinate mutation.
    saved_sphere = sphere_map.pose(zero_sphere)
    node = int(sphere_map.corner_nodes[0])
    x, y, z = np.asarray(
        sphere_model.mesh.get_node(node).coords(), dtype=float)
    sphere_model.mesh.set_node_coordinates(node, x + 1.0e-3, y, z)
    try:
        with pytest.raises(CurvedShellSkinError) as excinfo:
            sphere_map.spin_pullback(saved_sphere, np.zeros((len(sphere_map.dofs), 3)))
        assert excinfo.value.code == "mutated_bound_geometry"
    finally:
        sphere_model.mesh.set_node_coordinates(node, x, y, z)
    sphere_map.spin_pullback(saved_sphere, np.zeros((len(sphere_map.dofs), 3)))
    # A changed S3 reference director (facet normal) refuses, restore.
    face = sphere_map.faces[0]
    element = sphere_model.mesh.elements[face.element_id]
    normal = np.asarray(element.reference_normal, dtype=float).copy()
    element.reference_normal = -normal
    try:
        with pytest.raises(CurvedShellSkinError) as excinfo:
            sphere_map.pose(zero_sphere)
        assert excinfo.value.code == "mutated_bound_geometry"
    finally:
        element.reference_normal = normal
    sphere_map.pose(zero_sphere)


def test_face_interpolation_and_normals(sphere, panel):
    """Original interpolation/normals: S3 linear, Q4 bilinear on warped faces.

    Expected points and normals are computed independently from the libm
    exponential corner oracle and shape functions written here, not from the
    map code.  The warped-panel normals vary over the face, so they are
    current-geometry normals rather than rotated element directors.
    """
    sphere_model, sphere_map = sphere
    panel_model, panel_map, displacements = panel

    # S3: interior barycentric point, convention-independent centre and the
    # constant facet normal, all under a finite rotation.
    face_index = 0
    start, end = sphere_map.face_slices[face_index]
    theta = 0.05 * np.array((1.0, 2.0, 3.0)) / math.sqrt(14.0)
    q = _uniform_rotation(sphere_model, sphere_map, theta)
    pose = sphere_map.pose(q)
    independent = _independent_positions(sphere_model, sphere_map, q)
    weights = np.array((0.2, 0.3, 0.5))
    np.testing.assert_allclose(
        sphere_map.face_point(pose, face_index, weights),
        weights @ independent[start:end], rtol=0, atol=DIRECTOR_TOLERANCE)
    np.testing.assert_allclose(
        sphere_map.face_point(pose, face_index, np.full(3, 1.0 / 3.0)),
        independent[start:end].mean(axis=0), rtol=0, atol=1.0e-15)
    expected_normal = np.cross(independent[start + 1] - independent[start],
                               independent[start + 2] - independent[start])
    expected_normal /= np.linalg.norm(expected_normal)
    np.testing.assert_allclose(
        sphere_map.face_normal(pose, face_index, weights), expected_normal,
        rtol=0, atol=1.0e-12)

    # Q4: bilinear interior point and normal on a genuinely warped (locally
    # deformed, gravity-preloaded) face, plus the interior-force pullback.
    face_index = 0
    start, end = panel_map.face_slices[face_index]
    q_warp = displacements + _local_probe_field(panel_model, panel_map, seed=5)
    pose = panel_map.pose(q_warp)
    independent = _independent_positions(panel_model, panel_map, q_warp)
    xi, eta = (0.3, -0.2)
    weights = 0.25 * np.array((
        (1.0 - xi) * (1.0 - eta), (1.0 + xi) * (1.0 - eta),
        (1.0 + xi) * (1.0 + eta), (1.0 - xi) * (1.0 + eta)))
    np.testing.assert_allclose(
        panel_map.face_point(pose, face_index, (xi, eta)),
        weights @ independent[start:end], rtol=0, atol=DIRECTOR_TOLERANCE)
    np.testing.assert_allclose(
        panel_map.face_point(pose, face_index, (0.0, 0.0)),
        independent[start:end].mean(axis=0), rtol=0, atol=1.0e-15)
    d_xi = 0.25 * np.array((-(1.0 - eta), (1.0 - eta),
                            (1.0 + eta), -(1.0 + eta)))
    d_eta = 0.25 * np.array((-(1.0 - xi), -(1.0 + xi),
                             (1.0 + xi), (1.0 - xi)))
    expected_normal = np.cross(d_xi @ independent[start:end],
                               d_eta @ independent[start:end])
    expected_normal /= np.linalg.norm(expected_normal)
    np.testing.assert_allclose(
        panel_map.face_normal(pose, face_index, (xi, eta)), expected_normal,
        rtol=0, atol=1.0e-12)
    other = panel_map.face_normal(pose, face_index, (-0.4, 0.5))
    assert float(np.linalg.norm(other - expected_normal)) > 1.0e-6
    force = np.array((0.3, -0.7, 0.65))
    force /= np.linalg.norm(force)
    pulled = panel_map.face_force_pullback(pose, face_index, (xi, eta), force)
    independent_pullback = np.zeros(panel_map.n_dof)
    for local in range(4):
        corner = start + local
        dofs = panel_map.dofs[corner]
        rotated = rotation_matrix_from_vector(
            q_warp[dofs[3:]]) @ panel_map.directors[corner]
        independent_pullback[dofs[:3]] += weights[local] * force
        independent_pullback[dofs[3:]] += panel_map.offsets[corner] * np.cross(
            rotated, weights[local] * force)
    np.testing.assert_allclose(
        pulled, independent_pullback, rtol=0, atol=FORCE_PULLBACK_TOLERANCE)

    # Out-of-domain natural coordinates refuse.
    with pytest.raises(CurvedShellSkinError):
        panel_map.face_point(pose, face_index, (1.5, 0.0))
    with pytest.raises(CurvedShellSkinError):
        sphere_map.face_point(sphere_map.pose(np.zeros(sphere_map.n_dof)),
                              face_index, (0.5, 0.5, 0.5))
    # Barycentric weights that sum to one but leave the triangle refuse:
    # a sum-only check would silently admit extrapolation.
    with pytest.raises(CurvedShellSkinError):
        sphere_map.face_point(sphere_map.pose(np.zeros(sphere_map.n_dof)),
                              face_index, (1.5, -0.2, -0.3))
    with pytest.raises(CurvedShellSkinError):
        panel_map.face_point(pose, len(panel_map.faces), (0.0, 0.0))
