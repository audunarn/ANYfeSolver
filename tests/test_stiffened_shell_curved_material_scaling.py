"""Native retained-role normalization and physical similitude, not unit metadata."""
from dataclasses import replace
import numpy as np
import pytest
from anyfesolver import stiffened_shell as ss,b3_ge
from test_stiffened_shell_curved_material import (
    curved_material_panel,make_owner,_actions,assert_generalized_records_equal)
from test_stiffened_shell_curved_material_arc import control_for

NATIVE='native_force_moment_roles_v1'


def similar(d,g):
    """Scale all lengths by g and stresses by g^-2, keeping applied forces."""
    curve=d.shell.j2_curve
    curve=replace(curve,**{n:getattr(curve,n)/g**2 for n in
                         ('sigma_prop','sigma_yield','sigma_yield_2','K')})
    shell=replace(d.shell,coordinates_m=np.asarray(d.shell.coordinates_m)*g,
        thickness_m=d.shell.thickness_m*g,young_pa=d.shell.young_pa/g**2,
        density_kg_m3=d.shell.density_kg_m3/g**3,j2_curve=curve)
    D=np.diag([1.,1.,1.,g,g,g]);inv=np.diag([1.,1.,1.,1/g,1/g,1/g]);beams=[]
    for definition in d.beams:
        e,inertia=definition.instantiate();section=e.section
        law=b3_ge.EllipsoidalGeneralizedSection(D@section.elastic@D,
            inv@section.metric@inv,section.yield_force,section.hardening)
        beams.append(b3_ge.define_beam('b3-ge',e.element_id,e.node_ids,
            e.operator.reference.coordinates*g,e.operator.reference.nodal_triads,
            law,D@inertia@D/g,quadrature=e.operator.order,
            arithmetic_policy=e.operator.arithmetic_policy))
    return replace(d,shell=shell,beams=tuple(beams))


def route(owner):
    owner.solve(_actions(),(.5,1.),max_iterations=20,max_backtracks=12)
    control=control_for(owner,-1)
    first=owner.continue_arc_length(_actions(),control)
    prefix=owner.checkpoint()
    second=owner.continue_arc_length(_actions(),control)
    return first,prefix,second


@pytest.fixture(scope='module')
def baseline(request):
    d=replace(curved_material_panel(),retained_scale_policy=NATIVE)
    owner=make_owner(request,d)
    return d,owner,route(owner)


@pytest.mark.parametrize('g',[.1,1.,10.])
def test_native_scale_similarity_and_restart(request,baseline,g,tmp_path):
    definition,reference,base=baseline
    if g==1.:
        owner=reference;first,prefix,second=base
    else:
        owner=make_owner(request,similar(definition,g));first,prefix,second=route(owner)
    physical=6*(len(owner._shell_ids)+len(owner._beam_ids))
    expected=np.r_[np.tile([g,g,g,1.,1.,1.],physical//6),
                   np.tile(np.r_[np.ones(12),np.full(12,g)],len(owner._beam_parts))]
    np.testing.assert_allclose(owner._coordinate_scale/reference._coordinate_scale,
                               expected,rtol=1e-14,atol=0.)
    for actual,matched in ((first,base[0]),(second,base[2])):
        shell=actual.shell_displacement.reshape(-1,6).copy();shell[:,:3]/=g
        np.testing.assert_allclose(shell,matched.shell_displacement.reshape(-1,6),rtol=1e-7,atol=1e-11)
        forces=actual.cell_resultants.copy();forces[:,6:]/=g
        np.testing.assert_allclose(forces,matched.cell_resultants,rtol=1e-7,atol=1e-7)
        np.testing.assert_allclose(actual.shell_layer_alpha,matched.shell_layer_alpha,rtol=1e-7,atol=1e-12)
        np.testing.assert_allclose(actual.beam_station_accumulated,matched.beam_station_accumulated,rtol=1e-7,atol=1e-12)
        for a,b in zip(actual.shell_q4_material,matched.shell_q4_material):
            np.testing.assert_allclose(a.layer_stress*g**2,b.layer_stress,rtol=1e-7,atol=1e-5)
        assert abs(actual.amplitude-matched.amplitude)<1e-9
        assert actual.free_equilibrium_ratio<=1e-8 and actual.constraint_max<=1e-9
    fresh=make_owner(request,owner._definition);restored=fresh.restore(prefix)
    assert fresh.checkpoint()==prefix
    assert_generalized_records_equal(restored.beam_generalized_material,first.beam_generalized_material)
    repeated=fresh.continue_arc_length(_actions(),control_for(fresh,-1))
    assert owner.checkpoint()==fresh.checkpoint()
    assert_generalized_records_equal(second.beam_generalized_material,repeated.beam_generalized_material)
    np.testing.assert_array_equal(second.shell_layer_alpha,repeated.shell_layer_alpha)
    # The scale policy binds both force and arc snapshots to their compiled owner.
    old=make_owner(request,replace(owner._definition,retained_scale_policy='profile'))
    with pytest.raises(ValueError,match='metadata'):old.restore(prefix)
    assert old._generation==0 and old._finite_driver is None
    import json
    (tmp_path/'native-scale.json').write_text(json.dumps(dict(length_factor=g,
        arc_history=second.history,amplitude=second.amplitude,
        equilibrium=second.free_equilibrium_ratio,constraint=second.constraint_max,
        exact_restart=True),indent=2))


def test_native_policy_identity_admission_and_reference(request):
    from test_stiffened_shell_ge_experimental import _panel
    old=make_owner(request,similar(curved_material_panel(),1.))
    assert old._identity=='fd8cad3da68ac2b3ffff6bf8b0d55f75bf50eaecda2a30ff5e81f09227e6d453'
    assert old.capabilities['retained_coordinate_scale_policy']=='legacy_repeated_wrench_v1'
    d=replace(old._definition,retained_scale_policy=NATIVE);new=make_owner(request,d)
    assert new._identity!=old._identity
    assert new.capabilities['retained_coordinate_scale_policy']==NATIVE
    for value in ('unsupported',None,True):
        with pytest.raises(ValueError,match='native retained scaling'):
            ss.create_analysis(definition=replace(d,retained_scale_policy=value))
    with pytest.raises(ValueError,match='native retained scaling'):
        ss.create_analysis(definition=replace(_panel(),retained_scale_policy=NATIVE))
    new.solve(_actions(),(.5,),max_iterations=20,max_backtracks=12)
    prefix=new.checkpoint()
    with pytest.raises(ValueError,match='incompatible experimental binary checkpoint'):old.restore(prefix)
    assert old._generation==0 and old._finite_driver is None
    changes=(tuple(ss.ReferencePosePerturbation('shell',n,(.01,0.,0.)) for n in d.shell.node_ids)
             +tuple(ss.ReferencePosePerturbation('beam',n,(.01,0.,0.)) for n in (102,104)))
    ref=ss.create_stress_free_point_reference(d,changes)
    assert ref.definition.retained_scale_policy==NATIVE and ref.stress_free_scaled_max<=1e-9
    assert new.checkpoint()==prefix
