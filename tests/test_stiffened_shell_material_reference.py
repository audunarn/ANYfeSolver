"""Virgin material references are new models, not transported loaded histories."""
from dataclasses import replace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from anyfesolver import b3_ge,stiffened_shell as ss
from anyfesolver._stiffened_shell_ge_finite_experimental import FinitePointDriver
from test_stiffened_shell_curved_material import (
    curved_material_panel,make_owner,_actions,assert_generalized_records_equal,
)
from docs.reference_cases.stiffened_shell_e04_roof import build,definition,crown_reference_perturbations


def virgin_response(request,definition):
    owner=make_owner(request,definition);driver=FinitePointDriver(owner)
    driver._pattern=np.zeros(owner._K.shape[0])
    residual,_,_,gap,shell,beam=driver._assemble(driver._state,0.,driver._shell_origins,driver._beam_origins)
    assert np.max(abs(residual),initial=0.)<=1e-8 and gap<=1e-9
    for state in shell:
        np.testing.assert_array_equal(state['plastic_strain'],0.)
        np.testing.assert_array_equal(state['alpha'],0.)
        np.testing.assert_allclose(state['layer_stress'],0.,atol=1e-8)
    for history in beam:
        for station in history.stations:
            np.testing.assert_array_equal(station.plastic,0.)
            assert station.accumulated==(0.,0.)
    assert owner._generation==0 and owner._finite_driver is None
    return owner


def rigid_rows(d,motion,translation):
    rows=tuple(ss.ReferencePosePerturbation('shell',n,motion.apply(p)+translation-p,motion.as_rotvec())
        for n,p in zip(d.shell.node_ids,d.shell.coordinates_m))
    tied={p.beam_node for p in d.points};free={}
    for beam in d.beams:
        element,_=beam.instantiate()
        for n,p in zip(element.node_ids,element.operator.reference.coordinates):
            if n not in tied:free[n]=p
    return rows+tuple(ss.ReferencePosePerturbation('beam',n,motion.apply(p)+translation-p,
        motion.as_rotvec()) for n,p in free.items())


def test_virgin_material_reference_rigid_covariance_and_loaded_owner_isolation(request):
    d=curved_material_panel();source=make_owner(request,d)
    original=source.solve(_actions(),(.5,1.),max_iterations=20,max_backtracks=16)
    assert max(max(v) for v in original.shell_layer_alpha)>0.
    assert max(max(v) for v in original.beam_station_accumulated)>0.
    prefix=source.checkpoint();raw=tuple(v.raw for v in d.beams)
    motion=Rotation.from_rotvec((.2,-.1,.3));translation=np.array((.01,-.02,.03))
    reference=ss.create_stress_free_point_reference(d,rigid_rows(d,motion,translation))
    assert 'MATERIAL_REFERENCE_TRANSPORT' in reference.policy
    assert reference.source_identity==source._identity and reference.target_identity!=source._identity
    assert reference.stress_free_scaled_max<=1e-9 and not reference.production_qualified
    target=virgin_response(request,reference.definition)
    force=motion.apply(_actions()[0].wrench[:3])
    actions=(ss.NodalAction('beam',105,tuple(force)+(0.,0.,0.)),)
    result=target.solve(actions,(.5,1.),max_iterations=20,max_backtracks=16)
    expected=motion.apply(original.shell_displacement.reshape(-1,6)[:,:3].copy())
    np.testing.assert_allclose(result.shell_displacement.reshape(-1,6)[:,:3],expected,rtol=1e-9,atol=1e-10)
    np.testing.assert_allclose(result.beam_positions,motion.apply(original.beam_positions.copy())+translation,rtol=1e-9,atol=1e-10)
    np.testing.assert_allclose(result.beam_station_accumulated,original.beam_station_accumulated,rtol=1e-9,atol=1e-12)
    for a,b in zip(result.shell_q4_material,original.shell_q4_material):
        np.testing.assert_allclose(a.alpha,b.alpha,rtol=1e-9,atol=1e-12)
    assert source.checkpoint()==prefix and tuple(v.raw for v in d.beams)==raw
    with pytest.raises(ValueError,match='reference definition'):
        ss.create_stress_free_point_reference(source,())


@pytest.mark.parametrize('amplitude',(-1e-4,1e-4))
def test_actual_roof_material_crown_is_stress_free(request,amplitude):
    case=build(4,4);d=definition(case,density_kg_m3=1.)
    law=curved_material_panel().shell.j2_curve;beams=[]
    for old in d.beams:
        element,inertia=old.instantiate()
        section=b3_ge.EllipsoidalGeneralizedSection(element.section.elastic,element.section.metric,4000.,1e8)
        rebuilt=type(element)(element.element_id,element.node_ids,element.operator.reference,
            section,order=element.operator.order,arithmetic_policy=element.operator.arithmetic_policy)
        beams.append(type(old).capture(rebuilt,inertia))
    d=replace(d,shell=replace(d.shell,j2_curve=law),beams=tuple(beams),
        shell_reference_policy=ss.CURVED_Q4_POINT_MATERIAL_REFERENCE_V1)
    changed=ss.create_stress_free_point_reference(d,crown_reference_perturbations(case,amplitude))
    assert changed.source_identity!=changed.target_identity and changed.stress_free_scaled_max<=1e-9
    assert changed.definition.shell.j2_curve.as_dict()==law.as_dict()
    virgin_response(request,changed.definition)


def test_nonrigid_material_reference_yields_and_restarts(request):
    d=curved_material_panel()
    # Change shell width slope; lower-edge point hosts and native beam axes stay
    # fixed. Explicit zero port rotations retain the declared beam rest frames.
    rows=tuple(ss.ReferencePosePerturbation('shell',n,(0.,0.,1e-4*(p[1]+.5)))
        for n,p in zip(d.shell.node_ids,d.shell.coordinates_m))
    reference=ss.create_stress_free_point_reference(d,rows)
    assert reference.source_identity!=reference.target_identity
    owner=virgin_response(request,reference.definition)
    result=owner.solve(_actions(),(.5,1.),max_iterations=20,max_backtracks=16)
    assert max(max(v) for v in result.shell_layer_alpha)>0.
    assert max(max(v) for v in result.beam_station_accumulated)>0.
    fresh=make_owner(request,reference.definition);restored=fresh.restore(owner.checkpoint())
    assert fresh.checkpoint()==owner.checkpoint()
    assert_generalized_records_equal(result.beam_generalized_material,restored.beam_generalized_material)
    a=owner.solve(_actions(),(.9,),max_iterations=20,max_backtracks=16)
    b=fresh.solve(_actions(),(.9,),max_iterations=20,max_backtracks=16)
    assert owner.checkpoint()==fresh.checkpoint()
    assert_generalized_records_equal(a.beam_generalized_material,b.beam_generalized_material)
