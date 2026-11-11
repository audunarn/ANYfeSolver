"""Owned physical B2 definition and objective reference-frame oracles."""

from dataclasses import replace

import numpy as np
import pytest

import anyfesolver._b2_physical_linear_definition as definition_module
from anyfesolver._b2_physical_flexibility_linear import LinearSection
from anyfesolver._b2_physical_linear_definition import (
    GeometricMetadata, MassMetadata, PhysicalB2Member,
    PhysicalB2OperatorCache, PROFILE_ID,
)


def section():
    return LinearSection.isotropic(E=200., nu=.25, area=.4,
                                   Iy=.03, Iz=.03, J=.06, ky=.8, kz=.8)


def member(p0=(0., 0., 0.), p1=(2., 0., 0.), y=(0., 1., 0.), **metadata):
    return PhysicalB2Member("beam-A", "geometry/beam-A", (11, 27),
                            (p0, p1), y, "section-A", section(), **metadata)


def global_response(beam, displacement):
    transform=beam.local_dof_map()
    local=beam.local_operator().evaluate(transform@displacement)
    return local.energy, transform.T@local.force, transform.T@local.tangent@transform


def test_owned_reference_and_separate_mass_geometry_metadata():
    points=np.array([[0.,0.,0.],[2.,0.,0.]])
    y=np.array([0.,1.,0.])
    inertia=np.diag([.8,.8,.8,.02,.03,.04])
    mass=MassMetadata("mass-source",(0.,0.,0.),inertia)
    geometric=GeometricMetadata("shape-source",(0.,0.,0.),
                                (0.,.1,0.),(0.,-.1,0.),.2)
    beam=member(points[0],points[1],y,mass=mass,geometric=geometric)
    points[1,0]=9.;y[1]=4.;inertia[0,0]=99.
    assert beam.reference_points_m == ((0.,0.,0.),(2.,0.,0.))
    assert beam.local_y_global == (0.,1.,0.)
    assert mass.matrix_per_m[0][0] == .8
    assert beam.length_m == 2.
    assert beam.mass.source_id == "mass-source"
    assert beam.geometric.centroid_local_m == (0.,.1,0.)
    assert member().mass is None and member().geometric is None
    assert beam.profile_id == PROFILE_ID
    assert beam.unit_system == "SI"
    frame=beam.local_frame_rows();frame[0,0]=99.
    assert beam.local_frame_rows()[0,0] == 1.
    assert GeometricMetadata("no-Wagner",(0.,0.,0.),
                             (0.,.1,0.),(0.,-.1,0.)).geometric_polar_radius_squared_m2 is None


def test_mass_origin_pullback_matches_independent_particle_kinetic_work():
    points=np.array([(0.,.25+y,-.125+z) for y in (-.3,.3)
                     for z in (-.4,.4)])
    velocity=np.array([.13,-.21,.31,-.17,.29,.11])
    def kinetic_matrix(origin):
        matrix=np.zeros((6,6))
        for point in points:
            x,y,z=point-origin
            map_velocity=np.array([[1.,0.,0.,0.,z,-y],
                                   [0.,1.,0.,-z,0.,x],
                                   [0.,0.,1.,y,-x,0.]])
            matrix+=2.*map_velocity.T@map_velocity
        return matrix
    direct=kinetic_matrix(np.zeros(3))
    direct_energy=sum(np.linalg.norm(velocity[:3]+
                     np.cross(velocity[3:],point))**2 for point in points)
    assert .5*velocity@direct@velocity == pytest.approx(direct_energy,rel=1e-14)
    for origin in (np.array([0.,.1,.05]),np.array([0.,-.2,.3])):
        metadata=MassMetadata("particle-inertia",tuple(origin),
                              kinetic_matrix(origin))
        matrix=np.asarray(metadata.matrix_per_m)
        mass=matrix[0,0]
        coupling=matrix[:3,3:]
        center=np.array((coupling[1,2],coupling[2,0],
                         coupling[0,1]))/mass
        expected_center=points.mean(axis=0)-origin
        np.testing.assert_allclose(center,expected_center,rtol=2e-15,
                                   atol=2e-15)
        central=matrix[3:,3:]-mass*((center@center)*np.eye(3)-
                                     np.outer(center,center))
        recovered_second=.5*np.trace(central)*np.eye(3)-central
        direct_second=sum(2.*np.outer(p-points.mean(axis=0),
                                        p-points.mean(axis=0)) for p in points)
        np.testing.assert_allclose(recovered_second,direct_second,
                                   rtol=2e-14,atol=2e-14)
        pulled=metadata.matrix_at_reference_axis()
        np.testing.assert_allclose(pulled,direct,rtol=2e-15,atol=2e-15)
        assert .5*velocity@pulled@velocity == pytest.approx(direct_energy,rel=1e-14)
        pulled[0,0]=0.
        assert metadata.matrix_at_reference_axis()[0,0] == 8.


def test_compiled_member_reuses_one_operator_with_owned_outputs(monkeypatch):
    mass=MassMetadata("explicit-mass",(0.,.1,-.05),
                      np.diag([2.,2.,2.,.2,.3,.4]))
    beam=member(mass=mass)
    u=np.sin(np.arange(12)+.3)*.001
    stations=(.125,.4,.9)
    offset=(0.,.02,-.03)
    reference=beam.trace_global(u,stations=stations,offset_local_m=offset)
    reference_mass=beam.local_candidate_mass_matrix()
    actual_operator=definition_module.LinearOperator
    constructions=[]
    def counted_operator(*args,**kwargs):
        constructions.append(1)
        return actual_operator(*args,**kwargs)
    monkeypatch.setattr(definition_module,"LinearOperator",counted_operator)
    compiled=beam.compile_local()
    assert len(constructions)==1
    assert compiled.definition is beam and not compiled.production_qualified
    for _ in range(3):
        trace=compiled.trace_global(u,stations=stations,offset_local_m=offset)
        np.testing.assert_allclose(trace.reference_points_m,
                                   reference.reference_points_m,rtol=0.,atol=0.)
        np.testing.assert_allclose(trace.pose,reference.pose,rtol=0.,atol=0.)
        np.testing.assert_allclose(trace.jacobian,reference.jacobian,rtol=0.,atol=0.)
        np.testing.assert_allclose(compiled.local_candidate_mass_matrix(),
                                   reference_mass,rtol=0.,atol=0.)
    assert len(constructions)==1
    trace.pose[0,0]=99.
    trace.jacobian[0,0,0]=99.
    frame=compiled.local_frame_rows();frame[0,0]=99.
    assert compiled.trace_global(u,stations=stations,
                                 offset_local_m=offset).pose[0,0]!=99.
    assert compiled.local_frame_rows()[0,0]==1.
    changed=replace(beam,reference_points_m=((0.,0.,0.),(3.,0.,0.)))
    replacement=changed.compile_local()
    assert len(constructions)==2 and replacement.definition is changed
    assert replacement.operator is not compiled.operator
    assert replacement.definition.length_m!=compiled.definition.length_m
    with pytest.raises(ValueError,match="operator/definition mismatch"):
        definition_module.CompiledPhysicalB2Member(
            changed,compiled.operator,compiled._frame_bytes,
            compiled._transform_bytes)


def test_local_operator_cache_reuses_shared_section_and_length_only():
    shared_section = section()
    first = replace(member(), section=shared_section)
    turned = replace(member((0., 0., 0.), (0., 2., 0.), (-1., 0., 0.)),
                     member_id="beam-B", section=shared_section)
    longer = replace(first, member_id="beam-C",
                     reference_points_m=((0., 0., 0.), (3., 0., 0.)))
    equal_but_distinct = replace(first, member_id="beam-D", section=section())
    cache = PhysicalB2OperatorCache()
    compiled = [beam.compile_local(operator_cache=cache)
                for beam in (first, turned, longer, equal_but_distinct)]
    assert (cache.hits, cache.misses) == (1, 3)
    assert compiled[0].operator is compiled[1].operator
    assert compiled[0].operator is not compiled[2].operator
    assert compiled[0].operator is not compiled[3].operator
    assert not np.array_equal(compiled[0].local_frame_rows(),
                              compiled[1].local_frame_rows())
    displacement = np.sin(np.arange(12))*.001
    for beam, owned in zip((first, turned, longer, equal_but_distinct), compiled):
        fresh = beam.compile_local()
        np.testing.assert_array_equal(
            owned.trace_global(displacement).pose,
            fresh.trace_global(displacement).pose)
        np.testing.assert_array_equal(
            owned.operator.evaluate(beam.local_dof_map()@displacement).tangent,
            fresh.operator.evaluate(beam.local_dof_map()@displacement).tangent)
    with pytest.raises(TypeError, match="operator cache"):
        first.compile_local(operator_cache={})
    cache.clear()
    assert first.compile_local(operator_cache=cache).operator is not compiled[0].operator
    assert np.isfinite(compiled[0].operator.evaluate(np.zeros(12)).energy)


def test_admitted_roundoff_is_canonical_orthonormal_frame():
    beam=member(y=(5e-11,1.,0.))
    frame=beam.local_frame_rows()
    np.testing.assert_allclose(frame@frame.T,np.eye(3),rtol=0.,atol=1e-15)
    assert abs(beam.local_y_global[0]) < 1e-15


def test_reference_axis_rotation_and_reversed_connectivity_work():
    baseline=member()
    u=np.array([.1,-.2,.3,.04,-.02,.01,
                -.05,.07,-.08,.03,.02,-.04])
    energy,force,tangent=global_response(baseline,u)
    angle=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    rotate=np.kron(np.eye(4),angle)
    rotated=member((0.,0.,0.),(0.,2.,0.),(-1.,0.,0.))
    energy_rot,force_rot,tangent_rot=global_response(rotated,rotate@u)
    assert energy_rot == pytest.approx(energy,rel=1e-14)
    np.testing.assert_allclose(force_rot,rotate@force,rtol=1e-13,atol=1e-13)
    np.testing.assert_allclose(tangent_rot,rotate@tangent@rotate.T,
                               rtol=1e-13,atol=1e-13)

    swap=np.block([[np.zeros((6,6)),np.eye(6)],
                   [np.eye(6),np.zeros((6,6))]])
    reversed_beam=member((2.,0.,0.),(0.,0.,0.),(0.,1.,0.))
    energy_rev,force_rev,_=global_response(reversed_beam,swap@u)
    assert energy_rev == pytest.approx(energy,rel=1e-13)
    np.testing.assert_allclose(force_rev,swap@force,rtol=1e-12,atol=1e-12)


def test_generalized_reversal_requires_explicit_section_axis_transform():
    stiffness=np.diag([20.,11.,13.,7.,5.,9.])
    stiffness[0,1]=stiffness[1,0]=.8
    stiffness[2,4]=stiffness[4,2]=.3
    forward_section=LinearSection.generalized(stiffness)
    sign=np.diag([1.,-1.,1.,1.,-1.,1.])
    reversed_section=LinearSection.generalized(sign@stiffness@sign)
    common=("beam-A","source-A",(11,27))
    forward=PhysicalB2Member(*common,((0.,0.,0.),(2.,0.,0.)),
                              (0.,1.,0.),"section-A",forward_section)
    reverse=PhysicalB2Member(*common,((2.,0.,0.),(0.,0.,0.)),
                              (0.,1.,0.),"section-A/reoriented",reversed_section)
    unchanged=PhysicalB2Member(*common,((2.,0.,0.),(0.,0.,0.)),
                                (0.,1.,0.),"section-A",forward_section)
    u=np.array([.1,-.2,.3,.04,-.02,.01,
                -.05,.07,-.08,.03,.02,-.04])
    swap=np.block([[np.zeros((6,6)),np.eye(6)],
                   [np.eye(6),np.zeros((6,6))]])
    energy,force,_=global_response(forward,u)
    reversed_energy,reversed_force,_=global_response(reverse,swap@u)
    assert reversed_energy == pytest.approx(energy,rel=2e-14)
    np.testing.assert_allclose(reversed_force,swap@force,rtol=2e-13,atol=2e-13)
    assert abs(global_response(unchanged,swap@u)[0]-energy) > 1e-4


def test_precise_definition_negatives():
    with pytest.raises(ValueError,match="node IDs"):
        PhysicalB2Member("b","source",None,((0.,0.,0.),(1.,0.,0.)),
                         (0.,1.,0.),"s",section())
    with pytest.raises(ValueError,match="reference positions"):
        PhysicalB2Member("b","source",(1,2),None,
                         (0.,1.,0.),"s",section())
    with pytest.raises(ValueError,match="SI unit"):
        member(unit_system="inch-pound")
    with pytest.raises(ValueError,match="two distinct"):
        PhysicalB2Member("b","source",(1,1),((0.,0.,0.),(1.,0.,0.)),
                         (0.,1.,0.),"s",section())
    with pytest.raises(ValueError,match="perpendicular"):
        member(y=(1.,0.,0.))
    with pytest.raises(ValueError,match="nondegenerate"):
        member(p1=(0.,0.,0.))
    with pytest.raises(ValueError,match="profile/schema"):
        member(profile_id="historical-b2")
    with pytest.raises(ValueError,match="positive-semidefinite"):
        MassMetadata("m",(0.,0.,0.),np.diag([1.,1.,1.,1.,1.,-1.]))
    coupled=np.eye(6);coupled[0,1]=coupled[1,0]=2.
    with pytest.raises(ValueError,match="positive-semidefinite"):
        MassMetadata("m",(0.,0.,0.),coupled)
    with pytest.raises(ValueError,match="translational mass"):
        MassMetadata("anisotropic-translation",(0.,0.,0.),
                     np.diag([1.,2.,1.,1.,1.,1.]))
    nonphysical_first=np.eye(6)
    nonphysical_first[0,3]=nonphysical_first[3,0]=.1
    with pytest.raises(ValueError,match="first mass moment"):
        MassMetadata("non-skew-first-moment",(0.,0.,0.),nonphysical_first)
    with pytest.raises(ValueError,match="central inertia"):
        MassMetadata("impossible-principal-moments",(0.,0.,0.),
                     np.diag([1.,1.,1.,.1,.1,1.]))
    offset_without_valid_central=np.eye(6)
    offset_without_valid_central[0,5]=-1.
    offset_without_valid_central[5,0]=-1.
    offset_without_valid_central[2,3]=1.
    offset_without_valid_central[3,2]=1.
    with pytest.raises(ValueError,match="central inertia"):
        MassMetadata("impossible-offset-inertia",(0.,0.,0.),
                     offset_without_valid_central)
    distant=np.eye(6)
    distant[1,5]=distant[5,1]=1e7
    distant[2,4]=distant[4,2]=-1e7
    distant[3:,3:]=np.diag([0.,1e14,1e14])
    assert MassMetadata("distant-point",(0.,0.,0.),distant).matrix_per_m[4][4] == 1e14
    distant[3,3]=2.
    with pytest.raises(ValueError,match="central inertia"):
        MassMetadata("distant-impossible-central",(0.,0.,0.),distant)
    large_central=np.diag([1.,1.,1.,1e14,1e14,2e14+2.])
    with pytest.raises(ValueError,match="central inertia"):
        MassMetadata("large-impossible-central",(0.,0.,0.),large_central)
    massless_rotations=MassMetadata("translational",(0.,0.,0.),
                                    np.diag([1.,1.,1.,0.,0.,0.]))
    assert massless_rotations.matrix_per_m[5][5] == 0.
    with pytest.raises(ValueError,match="polar radius"):
        GeometricMetadata("g",(0.,0.,0.),(0.,0.,0.),(0.,0.,0.),0.)
