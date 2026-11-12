"""Reporting checks against an independent linearized physical-skin formula."""
import numpy as np
import pytest
from dataclasses import replace

from anyfesolver import ContactSkinGeometryError, recover_general_contact_shell_skins
from test_general_contact_q4_s3_patch_static import _case
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from anyfesolver import ContactDefinition, ContactSurface, create_shell_element


def _pose(model):
    u = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in range(1, 9):
        u[model.mesh.dof_manager.get_node_dofs(node)] = [node*1e-4, -node*2e-4,
            -node*1e-4, node*1e-3, -node*2e-3, node*3e-3]
    return u


def test_original_faces_and_independent_physical_skin_formula():
    model, definition, settings = _case()
    u = _pose(model)
    geometry = recover_general_contact_shell_skins(model, definition, settings, u)
    assert [face.node_ids for face in geometry.faces] == [(1,2,3,4), (5,6,8), (6,7,8)]
    assert not geometry.to_dict()['admitted_for_application']
    manager = GeneralS3PatchManager(model, definition, settings)
    owner = manager._positions(u)
    owner_nodes = (*manager.law.source_node_ids, *manager.law.target_node_ids)
    owner_lookup = dict(zip(owner_nodes, np.concatenate(owner)))
    for face in geometry.faces:
        offset = -.025 if face.role == 'source' else .015
        for index, node in enumerate(face.node_ids):
            dofs = model.mesh.dof_manager.get_node_dofs(node)
            point = model.mesh.nodes[node]
            reference = np.array([point.x,point.y,point.z]) + [0.,0.,offset]
            expected = reference + u[dofs[:3]] + offset*np.cross(u[dofs[3:]], [0.,0.,1.])
            np.testing.assert_allclose(face.reference_positions_m[index], reference, atol=1e-15)
            np.testing.assert_allclose(face.current_positions_m[index], expected, atol=1e-15)
            np.testing.assert_allclose(face.current_positions_m[index], owner_lookup[node], atol=1e-15)


def test_drill_and_common_translation_do_not_change_skin_motion():
    model, definition, settings = _case()
    u = _pose(model)
    base = recover_general_contact_shell_skins(model, definition, settings, u)
    drill = u.copy()
    for node in range(1,9): drill[model.mesh.dof_manager.get_node_dofs(node)[5]] += .2
    changed = recover_general_contact_shell_skins(model, definition, settings, drill)
    assert base == changed
    shift = np.array([3., -2., .7])
    moved_model, moved_definition, moved_settings = _case(shift)
    moved = recover_general_contact_shell_skins(moved_model, moved_definition, moved_settings, u)
    for a,b in zip(base.faces, moved.faces):
        assert a.node_ids == b.node_ids and a.side == b.side
        np.testing.assert_allclose(np.asarray(b.current_positions_m)-shift, a.current_positions_m, atol=1e-15)


def test_director_polarity_with_corresponding_side_and_offset():
    model, definition, settings = _case()
    base = recover_general_contact_shell_skins(model, definition, settings, _pose(model))
    flipped, _, _ = _case()
    for element_id, nodes in ((1,(1,2,3,4)), (2,(5,6,8)), (3,(6,7,8))):
        options = {} if element_id == 1 else dict(formulation='e4-pl-s3-v2d', reference_surface_offset=-.01)
        flipped.mesh.elements[element_id] = create_shell_element(element_id, nodes, 'elastic',
            thickness=.05, reference_normal=(0.,0.,-1.), **options)
    flipped_definition = ContactDefinition(
        surfaces=(ContactSurface('q4_source',(1,),'top'), ContactSurface('s3_target',(2,3),'bottom')),
        rigid_bodies=(), interactions=definition.interactions)
    changed = recover_general_contact_shell_skins(flipped, flipped_definition, settings, _pose(flipped))
    for a,b in zip(base.faces, changed.faces):
        np.testing.assert_allclose(a.reference_positions_m,b.reference_positions_m,atol=1e-15)
        np.testing.assert_allclose(a.current_positions_m,b.current_positions_m,atol=1e-15)


@pytest.mark.parametrize('kind', ['shape', 'bool', 'nan', 'inverted'])
def test_invalid_pose_fails_explicitly(kind):
    model, definition, settings = _case()
    u = np.zeros(model.mesh.dof_manager.total_dofs)
    if kind == 'shape': u = u[:-1]
    elif kind == 'bool': u = u.astype(bool)
    elif kind == 'nan': u[0] = np.nan
    else:
        for node in (2,3): u[model.mesh.dof_manager.get_node_dofs(node)[0]] = -2.
    with pytest.raises(ContactSkinGeometryError):
        recover_general_contact_shell_skins(model, definition, settings, u)


def test_publicly_unsupported_settings_do_not_receive_a_skin_profile():
    model, definition, settings = _case()
    settings = replace(settings, beam_shell_geometry='jet2_deformable_s3_dev')
    with pytest.raises(ContactSkinGeometryError, match='deformable beam-S3 selection'):
        recover_general_contact_shell_skins(model,definition,settings,np.zeros(model.mesh.dof_manager.total_dofs))
