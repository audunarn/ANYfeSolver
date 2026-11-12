"""Point-reference transport and lifecycle; full G4C remains open."""
from dataclasses import replace
import json
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken, SolveCancelled
from test_stiffened_shell_curved_reference_seam import curved_panel


def owner(request, definition):
    value = ss.create_analysis(definition=definition)
    request.addfinalizer(value.close)
    return value


def test_common_motion_transports_reference_and_offsets_once(request):
    source = curved_panel(); motion = Rotation.from_rotvec((.2,-.1,.3))
    translation = np.array((.01,-.02,.03))
    rows = tuple(ss.ReferencePosePerturbation('shell', n,
        motion.apply(p)+translation-p, motion.as_rotvec())
        for n,p in zip(source.shell.node_ids,source.shell.coordinates_m))
    tied={point.beam_node for point in source.points}
    free={}
    for beam in source.beams:
        element,_=beam.instantiate()
        for n,p in zip(element.node_ids,element.operator.reference.coordinates):
            if n not in tied:free[n]=p
    rows+=tuple(ss.ReferencePosePerturbation('beam',n,
        motion.apply(p)+translation-p,motion.as_rotvec()) for n,p in free.items())
    changed = ss.create_stress_free_point_reference(source,rows)
    assert changed.source_identity != changed.target_identity
    assert changed.stress_free_scaled_max <= 32*np.finfo(float).eps
    assert not changed.production_qualified
    original_owner = owner(request, source); target = owner(request, changed.definition)
    np.testing.assert_allclose(target._shell_xyz,
        original_owner._shell_xyz@motion.as_matrix().T+translation, atol=1e-15)
    for node in original_owner._beam_ids:
        p,r = original_owner._beam_nodes[node]; q,s = target._beam_nodes[node]
        # Unconnected stations are not inferred from shell geometry.
        if node not in {point.beam_node for point in source.points}:continue
        np.testing.assert_allclose(q,motion.as_matrix()@p+translation,atol=1e-15)
        np.testing.assert_allclose(s,motion.as_matrix()@r,atol=1e-15)
    for joint,old in zip(target._connections,original_owner._connections):
        np.testing.assert_allclose(joint._offset,motion.as_matrix()@old._offset,atol=1e-15)
        np.testing.assert_allclose(joint._relative,motion.as_matrix()@old._relative,atol=1e-15)
    # No second offset, section-axis correction, inertia change or tolerance
    # substitution enters native definition reconstruction.
    for before,after in zip(source.beams,changed.definition.beams):
        a,b=json.loads(before.raw),json.loads(after.raw)
        for key in ('section','section_inertia','quadrature','static_policy','dynamic_policy'):
            assert a[key]==b[key]
        for key in ('regularity_relative_tolerance','rotation_tolerance','frame_tolerance'):
            assert a['reference'][key]==b['reference'][key]


def test_zero_change_and_owned_pose_data():
    source = curved_panel()
    result = ss.create_stress_free_point_reference(source,())
    assert result.source_identity == result.target_identity
    assert tuple(b.raw for b in result.definition.beams)==tuple(b.raw for b in source.beams)
    values=np.array((.01,.02,.03));rotations=np.array((.1,.2,.3))
    row=ss.ReferencePosePerturbation('shell',1,values,rotations)
    values[:]=999;rotations[:]=999
    assert row.translation_m==(.01,.02,.03) and row.rotation_vector_rad==(.1,.2,.3)


def test_zero_change_material_reference_retains_material_policy():
    from anyfesolver._stiffened_shell_stress_free_reference import MATERIAL_POLICY
    from test_stiffened_shell_curved_material import curved_material_panel

    source = curved_material_panel()
    result = ss.create_stress_free_point_reference(source, ())
    assert result.policy == MATERIAL_POLICY
    assert result.definition.shell_reference_policy == source.shell_reference_policy
    assert result.source_identity == result.target_identity


@pytest.mark.parametrize('values', ((True,0.,0.), (float('nan'),0.,0.),
                                  (float('inf'),0.,0.), (0.,0.), ('1',0.,0.)))
def test_invalid_reference_components(values):
    with pytest.raises(ValueError):
        ss.ReferencePosePerturbation('shell',1,values)
    with pytest.raises(ValueError):
        ss.ReferencePosePerturbation('shell',1,(0.,0.,0.),values)


def test_multiple_hosts_and_shared_beam_pose(request,monkeypatch):
    source=curved_panel()
    source=replace(source,points=source.points+(ss.PointConnection(5,103),))
    owner(request,source)  # The starting constraint arrangement is admitted.
    translation=(0.,0.,.01)
    rows=tuple(ss.ReferencePosePerturbation('shell',n,translation)
               for n in source.shell.node_ids)
    rows+=tuple(ss.ReferencePosePerturbation('beam',n,translation) for n in (102,104))
    moved=ss.create_stress_free_point_reference(source,rows)
    assert moved.max_transport_disagreement==0.
    left,_=moved.definition.beams[0].instantiate()
    right,_=moved.definition.beams[1].instantiate()
    np.testing.assert_array_equal(left.operator.reference.coordinates[-1],
                                  right.operator.reference.coordinates[0])
    np.testing.assert_array_equal(left.operator.reference.nodal_triads[-1],
                                  right.operator.reference.nodal_triads[0])
    with pytest.raises(ValueError,match='multiple-host'):
        ss.create_stress_free_point_reference(source,
            (ss.ReferencePosePerturbation('shell',2,translation),))
    # Isolate the regression from geometry admission: unrelated model-wide
    # scale metadata cannot loosen local host arithmetic-roundoff agreement.
    owner_type=type(owner(request,source));preflight=owner_type.preflight
    def unrelated_scale(value):
        info=preflight(value);info['reference_length_m']=1e6;return info
    monkeypatch.setattr(owner_type,'preflight',unrelated_scale)
    with pytest.raises(ValueError,match='multiple-host'):
        ss.create_stress_free_point_reference(source,
            (ss.ReferencePosePerturbation('shell',2,(0.,0.,1e-9)),))


def test_cancel_after_target_compile_closes_both_owners(monkeypatch):
    source=curved_panel();created=[];actual=ss.create_analysis
    def tracked(**kwargs):
        value=actual(**kwargs);created.append(value);return value
    monkeypatch.setattr(ss,'create_analysis',tracked)
    from anyfesolver import _stiffened_shell_stress_free_reference as reference
    safe=reference.cancellation_safe_point
    token=CancellationToken()
    def cancelling(value,where):
        if where=='stiffened-shell.reference.complete':
            token.cancel('after compilation, before publication')
        return safe(value,where)
    monkeypatch.setattr(reference,'cancellation_safe_point',cancelling)
    rows=tuple(ss.ReferencePosePerturbation('shell',n,(0.,0.,.01))
               for n in source.shell.node_ids)
    rows+=tuple(ss.ReferencePosePerturbation('beam',n,(0.,0.,.01)) for n in (102,104))
    with pytest.raises(SolveCancelled):
        ss.create_stress_free_point_reference(source,rows,cancellation_token=token)
    assert len(created)==2
    for value in created:
        with pytest.raises(RuntimeError,match='closed'):
            value.preflight()


def test_native_policy_and_reference_tolerances_are_preserved():
    from anyfesolver._ge_beam3_precise_geometric_work import POLICY
    source=curved_panel();beams=[]
    for raw in source.beams:
        element,inertia=raw.instantiate();ref=element.operator.reference
        reference=type(ref)(ref.coordinates,ref.nodal_triads,
            regularity_relative_tolerance=2*ref._regularity_relative_tolerance,
            rotation_tolerance=.5*ref._rotation_tolerance,
            frame_tolerance=.5*ref._frame_tolerance)
        precise=type(element)(element.element_id,element.node_ids,reference,
            element.section,order=element.operator.order,arithmetic_policy=POLICY)
        beams.append(type(raw).capture(precise,inertia))
    source=replace(source,beams=tuple(beams))
    rows=tuple(ss.ReferencePosePerturbation('shell',n,(0.,0.,.01))
               for n in source.shell.node_ids)
    rows+=tuple(ss.ReferencePosePerturbation('beam',n,(0.,0.,.01)) for n in (102,104))
    changed=ss.create_stress_free_point_reference(source,rows)
    for old,new in zip(source.beams,changed.definition.beams):
        before,after=json.loads(old.raw),json.loads(new.raw)
        assert before['schema']==after['schema']
        assert before['arithmetic_policy']==after['arithmetic_policy']==POLICY
        for key in ('regularity_relative_tolerance','rotation_tolerance','frame_tolerance'):
            assert before['reference'][key]==after['reference'][key]


def test_admission_failures_preserve_the_original_reference():
    source=curved_panel();raw=tuple(b.raw for b in source.beams)
    with pytest.raises(ValueError,match='reference definition'):
        ss.create_stress_free_point_reference(replace(source,shell=None),())
    with pytest.raises(ValueError,match='point-connected'):
        ss.create_stress_free_point_reference(source,(ss.ReferencePosePerturbation('beam',101,(0.,0.,0.)),))
    row=ss.ReferencePosePerturbation('shell',1,(0.,0.,0.))
    with pytest.raises(ValueError,match='duplicate'):
        ss.create_stress_free_point_reference(source,(row,row))
    with pytest.raises(ValueError,match='unknown'):
        ss.create_stress_free_point_reference(source,(ss.ReferencePosePerturbation('shell',999,(0.,0.,0.)),))
    # A single lifted Q4 corner is not an admitted planar facet.
    with pytest.raises(ValueError,match='planar'):
        ss.create_stress_free_point_reference(source,(ss.ReferencePosePerturbation('shell',4,(0.,0.,.001)),))
    # The midpoint remains at its old position while endpoints move. Native
    # Q2 frame/centerline compatibility must reject, not silently rotate it.
    with pytest.raises(ValueError,match='tangent|frame|aligned'):
        ss.create_stress_free_point_reference(source,(ss.ReferencePosePerturbation('beam',102,(0.,0.,.001)),))
    assert tuple(b.raw for b in source.beams)==raw
    token=CancellationToken();token.cancel('before reference creation')
    with pytest.raises(SolveCancelled):
        ss.create_stress_free_point_reference(source,(),cancellation_token=token)


def roof_changes(case, amplitude):
    from docs.reference_cases.stiffened_shell_e04_roof import crown_reference_perturbations
    return crown_reference_perturbations(case,amplitude)


@pytest.mark.parametrize('amplitude',(-1e-4,1e-4))
def test_actual_roof_imperfection_is_virgin_and_restarts(request, amplitude):
    from docs.reference_cases.stiffened_shell_e04_roof import build,definition
    c=build(4,4);source=definition(c,density_kg_m3=1.)
    previous=owner(request,source)
    action=(ss.NodalAction('shell',c.loaded_shell_node+1,(0.,0.,-10.,0.,0.,0.)),)
    previous.solve(action,(1.,));accepted=previous.checkpoint()
    changed=ss.create_stress_free_point_reference(source,roof_changes(c,amplitude))
    assert previous.checkpoint()==accepted
    target=owner(request,changed.definition)
    assert target.preflight()['stress_free_scaled_max']<=32*np.finfo(float).eps
    assert target._generation==0 and target._finite_driver is None
    for joint in target._connections:
        assert np.linalg.norm(joint._offset)==pytest.approx((c.thickness+c.bar_height)/2,rel=1e-13)
    np.testing.assert_allclose(np.array(changed.definition.shell.coordinates_m)[c.loaded_shell_node,2],
        c.shell_positions[c.loaded_shell_node,2]-amplitude,atol=1e-16)
    with pytest.raises(ValueError):target.restore(accepted)
    result=target.solve(action,(1.,));saved=target.checkpoint()
    fresh=owner(request,changed.definition);fresh.restore(saved)
    actual=target.solve(action,(2.,));repeated=fresh.solve(action,(2.,))
    assert fresh.checkpoint()==target.checkpoint()
    np.testing.assert_array_equal(actual.cell_resultants,repeated.cell_resultants)
    assert actual.free_equilibrium_ratio<=1e-8 and actual.constraint_max<=1e-9
