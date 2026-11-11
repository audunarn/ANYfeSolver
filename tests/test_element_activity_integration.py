from __future__ import annotations

import numpy as np

from anyfesolver import (
    ContributionPolicy,
    ElementActivity,
    ElementActivityPolicy,
    LoadCase,
    assemble_damping_matrix,
    assemble_load_vector,
    assemble_mass_matrix,
    assemble_stiffness_matrix,
    generate_beam_mesh,
    generate_simple_panel_mesh,
)


def _all_contributions_policy() -> ElementActivityPolicy:
    return ElementActivityPolicy(
        stiffness=ContributionPolicy.ACTIVITY,
        mass=ContributionPolicy.ACTIVITY,
        damping=ContributionPolicy.ACTIVITY,
        load=ContributionPolicy.ACTIVITY,
        contact=ContributionPolicy.ACTIVITY,
    )


def test_activity_scales_local_matrices_and_invalidates_revision() -> None:
    model = generate_beam_mesh(
        1.0,
        num_divisions=1,
        cross_section={"area": 0.01, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6},
    )
    model.materials["steel"].density = 7850.0
    baseline_k, _ = assemble_stiffness_matrix(model)
    baseline_m, _ = assemble_mass_matrix(model)
    baseline_c, _ = assemble_damping_matrix(model, 0.2, 0.3)
    element_ids = tuple(model.mesh.elements)
    activity = ElementActivity(
        element_ids, 1.0, policy=_all_contributions_policy()
    )
    model.set_element_activity(activity)
    attached_revision = model.mesh.revision_signature()["activity"]

    activity.set_activity(element_ids, 0.25, reason="integration-test")
    stiffness, stiffness_info = assemble_stiffness_matrix(model)
    mass, mass_info = assemble_mass_matrix(model)
    damping, damping_info = assemble_damping_matrix(model, 0.2, 0.3)

    np.testing.assert_allclose(stiffness.toarray(), 0.25 * baseline_k.toarray())
    np.testing.assert_allclose(mass.toarray(), 0.25 * baseline_m.toarray())
    np.testing.assert_allclose(damping.toarray(), 0.25 * baseline_c.toarray())
    assert model.mesh.revision_signature()["activity"] > attached_revision
    assert stiffness_info["diagnostics"]["element_activity"]["scaled_element_count"] == 1
    assert mass_info["diagnostics"]["element_activity"]["quantity"] == "mass"
    assert damping_info["stiffness"]["diagnostics"]["element_activity"]["quantity"] == "damping"


def test_element_loads_scale_but_nodal_loads_remain_node_owned() -> None:
    model = generate_simple_panel_mesh(
        1.0, 1.0, 0.01, num_divisions_x=1, num_divisions_y=1
    )
    element_id = next(iter(model.mesh.elements))
    load = LoadCase("mixed")
    load.add_pressure_load(element_id, 1000.0)
    load.add_nodal_load(1, [5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    baseline, _ = assemble_load_vector(model, load)
    nodal = LoadCase("nodal")
    nodal.add_nodal_load(1, [5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    nodal_only, _ = assemble_load_vector(model, nodal)
    activity = ElementActivity(
        [element_id], 0.4, policy=_all_contributions_policy()
    )
    model.set_element_activity(activity)

    scaled, info = assemble_load_vector(model, load)

    np.testing.assert_allclose(scaled, nodal_only + 0.4 * (baseline - nodal_only))
    assert info["element_activity"]["sequence"] == activity.sequence


def test_hard_deletion_keeps_topology_but_removes_contributions() -> None:
    model = generate_beam_mesh(
        1.0,
        num_divisions=1,
        cross_section={"area": 0.01, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6},
    )
    element_ids = tuple(model.mesh.elements)
    activity = ElementActivity(
        element_ids, policy=_all_contributions_policy()
    )
    model.set_element_activity(activity)
    activity.hard_delete(element_ids)

    stiffness, info = assemble_stiffness_matrix(model)

    assert tuple(model.mesh.elements) == element_ids
    assert stiffness.nnz == 0
    assert info["diagnostics"]["element_activity"]["zero_contribution_count"] == 1
