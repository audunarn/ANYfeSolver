"""Independent normal-work oracle for skew, partially overlapping shell patches."""

import numpy as np
import pytest

from anyfesolver import (
    ContactDefinition, ContactInteraction, ContactSolverSettings,
    ContactSurface, create_shell_element,
)
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager


_TARGET = ((.2, -.1), (.92, 0.), (.86, 1.15), (.15, .95))


def _reference_polygon():
    """Clip a planar polygon against four scalar bounds, outside the solver."""
    polygon = _TARGET
    for coordinate, bound, above in ((0, .1, True), (0, 1.1, False),
                                     (1, .1, True), (1, 1.1, False)):
        def level(point):
            value = point[coordinate]-bound
            return value if above else -value

        clipped = []
        for first, second in zip(polygon[-1:]+polygon[:-1], polygon):
            a, b = level(first), level(second)
            if (a >= 0.) != (b >= 0.):
                fraction = a/(a-b)
                clipped.append(tuple(first[k]+fraction*(second[k]-first[k])
                                     for k in (0, 1)))
            if b >= 0.:
                clipped.append(second)
        polygon = tuple(clipped)
    return polygon


def _reference_wrench(pressure):
    polygon = _reference_polygon()
    twice_area = 0.
    first_x = 0.
    first_y = 0.
    for first, second in zip(polygon, polygon[1:]+polygon[:1]):
        cross = first[0]*second[1]-second[0]*first[1]
        twice_area += cross
        first_x += (first[0]+second[0])*cross
        first_y += (first[1]+second[1])*cross
    area = twice_area/2.
    centroid = (first_x/(3*twice_area), first_y/(3*twice_area))
    normal_force = pressure*area
    return area, np.array((0., 0., normal_force)), np.array((
        normal_force*centroid[1], -normal_force*centroid[0], 0.,
    ))


def _case(count):
    model = FEModel(f"skew_partial_q4_{count}")
    model.add_material("elastic", 1.e5, .3, density=1.)
    for row, y in enumerate((.1, 1.1)):
        for index in range(count+1):
            model.add_node(1+row*(count+1)+index,
                           .1+index/count, y, .055)
    first_target = 2*(count+1)+1
    for index, (x, y) in enumerate(_TARGET):
        model.add_node(first_target+index, x, y, 0.)
    for index in range(count):
        lower = 1+index
        upper = 1+count+1+index
        model.add_element(index+1, create_shell_element(
            index+1, (lower, lower+1, upper+1, upper),
            "elastic", thickness=.05,
            reference_normal=(0., 0., 1.),
        ))
    for element_id, nodes in (
            (count+1, (first_target, first_target+1, first_target+3)),
            (count+2, (first_target+1, first_target+2, first_target+3))):
        model.add_element(element_id, create_shell_element(
            element_id, nodes, "elastic", thickness=.05,
            formulation="e4-pl-s3-v2d",
            reference_normal=(0., 0., 1.),
            reference_surface_offset=.01,
        ))
    definition = ContactDefinition(
        surfaces=(ContactSurface("source", tuple(range(1, count+1)), "bottom"),
                  ContactSurface("target", (count+1, count+2), "top")),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "pair", "source", "target", 1000., .01,
            normal_law="augmented",
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="general", normal_discretization="primal_mortar",
        general_s3_difference_step=2.e-5,
        maximum_general_s3_energy_evaluations=14144,
        maximum_candidate_pairs=32,
        q4_source_area_tolerance=1.e-10,
        q4_source_weighted_gap_tolerance=1.e-11,
        maximum_q4_source_integration_triangles=500,
        pressure_tolerance=2.e-4,
    )
    manager = GeneralS3PatchManager(model, definition, settings)
    manager.set_normal_multipliers(np.ones(len(manager.law.source_node_ids)))
    displacement = np.zeros(manager.n_dof)
    for node_id in range(1, 2*(count+1)+1):
        displacement[model.mesh.dof_manager.get_node_dofs(node_id)[2]] = -.017
    return model, manager, displacement


@pytest.mark.parametrize("count, supported_count, unsupported_ids", (
    (2, 6, ()), (4, 10, ()), (8, 16, (9, 18)),
))
def test_skew_partial_uniform_gap_has_only_normal_work(
        count, supported_count, unsupported_ids):
    _model, manager, displacement = _case(count)
    _force, _tangent, result = manager.evaluate(displacement, tangent=False)
    area, expected_force, expected_moment = _reference_wrench(3.)
    assert result.reference_overlap_area == pytest.approx(area, abs=1.e-10)
    assert result.minimum_gap == pytest.approx(-.002, abs=1.e-10)
    assert result.pressure_gram_rank == supported_count
    assert sum(value > 0. for value in result.reference_support_m2) == supported_count
    assert tuple(node_id for node_id, support in zip(
        result.source_pressure_node_ids, result.reference_support_m2)
        if support == 0.) == unsupported_ids
    np.testing.assert_allclose(result.integration_force, expected_force,
                               rtol=0., atol=1.e-5)
    np.testing.assert_allclose(result.integration_moment, expected_moment,
                               rtol=0., atol=1.e-5)
    np.testing.assert_allclose(np.asarray(result.integration_force)+result.target_force,
                               0., rtol=0., atol=1.e-8)
    np.testing.assert_allclose(np.asarray(result.integration_moment)+result.target_moment,
                               0., rtol=0., atol=1.e-8)
    assert result.area_integration_error_m2 <= 1.e-10
    assert result.weighted_gap_integration_error_m3 <= 1.e-11


def test_skew_partial_normal_work_has_directional_tangent():
    model, manager, displacement = _case(2)
    _force, tangent, _result = manager.evaluate(displacement, tangent=True)
    direction = np.zeros(manager.n_dof)
    for node_id in range(1, 7):
        dofs = model.mesh.dof_manager.get_node_dofs(node_id)
        direction[dofs[1]] = .4
    direction[model.mesh.dof_manager.get_node_dofs(2)[2]] = .1
    direction[model.mesh.dof_manager.get_node_dofs(8)[2]] = -.05
    step = 1.e-4
    plus = manager.evaluate(displacement+step*direction, tangent=False)[0]
    minus = manager.evaluate(displacement-step*direction, tangent=False)[0]
    independent_directional = (plus-minus)/(2*step)
    predicted = tangent @ direction
    error = np.linalg.norm(independent_directional-predicted)/max(
        1., float(np.linalg.norm(independent_directional)),
    )
    assert error <= 1.e-5
