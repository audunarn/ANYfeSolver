"""Literal-radius and prescribed-path representation discriminators.

Fractions establish the declared binary-input geometry independently of the
production radial kernel. No beam response or broad capability claim is implied.
"""
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from anyfesolver.general_contact.beam_hemisphere_cylinder import HemisphereCylinderManager
from anyfesolver.general_contact.beam_hemisphere_cylinder_motion import TranslatingHemisphereCylinderManager
from test_general_contact_radial_beam_ends import _case


@pytest.mark.parametrize("beam_radius", (.01, .05))
def test_literal_radius_controls_line_activation_and_work(beam_radius):
    model, definition, settings = _case("cylinder")
    obstacle_radius = 1e7
    rounded = obstacle_radius + beam_radius
    initial_x = np.nextafter(rounded, np.inf)
    for node in model.mesh.nodes.values():
        # Parallel to the axis: a constant radial gap and exact unit measure.
        node.x = initial_x
        node.y = float(node.id - 1)
        node.z = 0.
    definition = replace(definition,
        surfaces=(replace(definition.surfaces[0], beam_radius=beam_radius),),
        rigid_bodies=(replace(definition.rigid_bodies[0], radius=obstacle_radius, normal=(0., 1., 0.)),))
    settings = replace(settings, beam_end_cap_work="hemisphere_support_dev")
    manager = HemisphereCylinderManager(model, definition, settings)
    cell = manager.cells[0]
    positions = np.array(((rounded, 0., 0.), (rounded, 1., 0.)))
    exact_gap = float(Fraction.from_float(rounded) - Fraction.from_float(obstacle_radius)
                      - Fraction.from_float(beam_radius))
    lanes = manager._axis_controls(positions, manager.point)
    actual_gap, _, _ = manager._radial_work(tuple((lane[0],) for lane in lanes))
    assert actual_gap == pytest.approx(exact_gap, rel=1e-14, abs=1e-24)
    energy, force, tangent, active, *_ = manager._integrate_cell(cell, positions)
    expected_gap = min(exact_gap, 0.)
    assert energy == pytest.approx(.5 * manager.stiffness * expected_gap**2 * cell.length, rel=1e-13, abs=1e-30)
    assert force[::3].sum() == pytest.approx(manager.stiffness * expected_gap * cell.length, rel=1e-13, abs=1e-20)
    assert active == pytest.approx(cell.length if exact_gap < 0. else 0.)
    assert np.all(np.isfinite(tangent))


def test_sub_ulp_prescribed_tool_point_preserves_literal_affine_motion():
    manager = object.__new__(TranslatingHemisphereCylinderManager)
    manager.reference_point = np.array((1e16, 0., 0.))
    manager.path_translation = np.array((1., 0., 0.))
    point=manager.point_at(1.)
    assert point[0]==Fraction(10**16)+1
    assert point[0]-manager.point_at(0.)[0]==1


def test_sub_ulp_motion_reaches_line_cap_sweep_and_rigid_work():
    model, definition, settings = _case("cylinder")
    origin=1e16
    for node in model.mesh.nodes.values():
        node.x=origin+4.;node.y=float(node.id-1);node.z=0.
    definition=replace(definition,
        surfaces=(replace(definition.surfaces[0],beam_radius=1.),),
        rigid_bodies=(replace(definition.rigid_bodies[0],point=(origin,0.,0.),radius=1.9,
                             normal=(0.,1.,0.),path_translation=(1.,0.,0.)),),
        interactions=(replace(definition.interactions[0],gap_tolerance=.8),))
    settings=replace(settings,beam_end_cap_work="hemisphere_support_dev")
    manager=TranslatingHemisphereCylinderManager(model,definition,settings)
    import json
    initial_report=json.loads(json.dumps(manager.tool_position_report(),allow_nan=False))
    assert initial_report['rigid_tool_position_rational_m'][0]==dict(numerator=str(10**16),denominator='1')
    u=np.zeros(model.mesh.dof_manager.total_dofs)
    energies=[]
    for factor in (1.2,1.5):
        manager.set_path_factor(factor)
        exact_gap=float(Fraction(4)-Fraction.from_float(factor)-Fraction(1)-Fraction.from_float(1.9))
        _,_,result=manager.evaluate(u)
        assert all(field['gap']==pytest.approx(exact_gap,rel=1e-14) for field in result.endpoint_fields)
        assert result.active_endpoints==2 and result.active_line_length==pytest.approx(1.)
        energies.append(.5*(manager.stiffness+2*manager.endpoint_stiffness)*exact_gap**2)
        assert result.line_contact_energy+result.endpoint_contact_energy==pytest.approx(energies[-1],rel=1e-13)
    previous=manager.path_factor
    assert manager._swept_admissible(u,u,1.2,1.5)
    assert manager.tool_work_increment(u,u,1.2,1.5)==pytest.approx(energies[0]-energies[1],rel=1e-13)
    assert manager.path_factor==previous
    assert manager.restart_contract()['prescribed_rigid_kinematics'].endswith('_v2')
    report=json.loads(json.dumps(manager.tool_position_report(),allow_nan=False))
    exact=report['rigid_tool_position_rational_m'][0]
    assert Fraction(int(exact['numerator']),int(exact['denominator']))==Fraction(10**16)+Fraction(3,2)
    assert report['rigid_tool_position_representation']=='float_display_with_exact_rational_m'


def test_sphere_line_gap_matches_literal_radius_and_legacy_remains_rounded():
    from anyfesolver.general_contact.beam_hemisphere_sphere import HemisphereSphereManager
    from anyfesolver.general_contact.beam_radial_ends import BeamRadialEndManager
    from anyfesolver.general_contact.beam_end_radial_geometry import radial_point_work
    radius=Fraction.from_float(1e7)+Fraction.from_float(.01)
    point=((Fraction.from_float(1e7+.01),),(Fraction(0),),(Fraction(0),))
    manager=object.__new__(HemisphereSphereManager)
    manager.full_envelope_radius=radius;manager.projector=np.eye(3)
    assert manager._radial_work(point)[0]==float(point[0][0]-radius)
    legacy=object.__new__(BeamRadialEndManager)
    legacy.radius=float(radius);legacy.projector=np.eye(3)
    assert legacy._radial_work(point)[0]==radial_point_work(point,np.eye(3),float(radius))[0]==0.


@pytest.mark.parametrize('reverse',(False,True))
def test_exact_radius_tiny_b3_endpoint_interval_is_not_hidden(reverse):
    model,definition,settings=_case('cylinder',True)
    manager=HemisphereCylinderManager(model,definition,replace(settings,beam_end_cap_work='hemisphere_support_dev'))
    positions=np.array(((.11,0.,0.),(.21,0.,0.),(.31,0.,0.)))
    if reverse:positions=positions[::-1].copy()
    exact_radius=Fraction.from_float(.1)+Fraction.from_float(.01)
    x0,xm,x1=(Fraction.from_float(x) for x in positions[:,0])
    # Independent exact Lagrange polynomial and a bracketed scalar front.
    a,b=x0,4*xm-3*x0-x1
    c=2*(x0+x1-2*xm)
    left,right=Fraction(0),Fraction(1)
    for _ in range(150):
        middle=(left+right)/2
        if (a+b*middle+c*middle*middle<exact_radius)!=reverse:left=middle
        else:right=middle
    root=(left+right)/2
    width=float(1-root if reverse else root)
    gap0=float((a+b+c if reverse else a)-exact_radius)
    def record(mode,**values):
        import json,os
        from pathlib import Path
        path=os.environ.get('CONTACT_APP_RESULT_RECORD')
        if path:Path(path).with_name(f'tiny_front_reverse_{reverse}.json').write_text(
            json.dumps(dict(mode=mode,reverse=reverse,reference_width=width,
                            reference_endpoint_gap_m=gap0,**values),indent=2),encoding='utf-8')
    try:E,f,K,active,*_=manager._integrate_cell(manager.cells[0],positions)
    except (ArithmeticError,MemoryError) as error:
        # This deliberately unrepresentable right-end front may fail closed;
        # no accepted response, dropped force or engineering pass is claimed.
        assert reverse and ('unresolved' in str(error) or 'budget' in str(error))
        record('explicit_non_success',exception=type(error).__name__,reason=str(error))
        return
    assert 0.<width<1e-14 and gap0<0.
    assert active==pytest.approx(manager.cells[0].length*width,rel=1e-6)
    assert f[::3].sum()==pytest.approx(.5*manager.stiffness*gap0*width*manager.cells[0].length,rel=1e-6,abs=1e-50)
    assert E==pytest.approx(manager.stiffness*gap0**2*width*manager.cells[0].length/6,rel=1e-6,abs=1e-65)
    assert np.all(np.isfinite(K))
    record('accepted_local_work',active_length_m=active,force_x_N=float(f[::3].sum()),energy_J=E)


def test_sub_ulp_curved_b3_active_rim_matches_local_geometry_and_formula():
    from anyfesolver.general_contact.beam_hemisphere_cylinder_work import hemisphere_cylinder_work
    origin=10**16
    local=np.array(((0.,0.,0.),(2.,0.,1.),(4.,0.,3.)))
    positions=local.copy();positions[:,0]+=origin
    exact_center=(Fraction(origin)+1,Fraction(0),Fraction(0))
    state=hemisphere_cylinder_work(positions,2,1,1.,exact_center,(0.,0.,1.),2.5)
    control=hemisphere_cylinder_work(local,2,1,1.,(1.,0.,0.),(0.,0.,1.),2.5)
    # Outward tangent=(4,0,5). The rim's maximum inward x component is
    # 5/sqrt(41); z is cylinder-invariant. This is an independent scalar law.
    expected=3.-5./np.sqrt(41.)-2.5
    assert state.branch==control.branch=='rim' and expected<0
    assert state.gap==pytest.approx(expected,abs=1e-12)
    np.testing.assert_allclose(state.gradient,control.gradient,rtol=1e-13,atol=1e-14)
    np.testing.assert_allclose(state.hessian,control.hessian,rtol=1e-13,atol=1e-14)


@pytest.mark.parametrize('moving',(False,True))
def test_pre_correction_restart_contract_is_rejected(moving,monkeypatch):
    import json
    from anyfesolver import LoadCase, solve_static_nonlinear
    from anyfesolver.general_contact.beam_hemisphere_sphere import HemisphereSphereManager
    from test_general_contact_coupled_tool_staging import coupled_fixture
    from test_general_contact_hemisphere_cylinder_static import _bar_case
    factory=coupled_fixture if moving else _bar_case
    model,definition,settings,_=factory()
    load=LoadCase('open contract checkpoint')
    common=dict(contact=definition,contact_settings=settings,num_steps=1,max_iterations=20,
                tolerance=1e-10,emit_restart_checkpoint=True)
    original_radius=HemisphereSphereManager.restart_contract
    original_path=TranslatingHemisphereCylinderManager.restart_contract
    def old_radius(self):
        contract=original_radius(self);contract.pop('line_radius_work_revision');return contract
    def old_path(self):
        contract=original_path(self);contract['prescribed_rigid_kinematics']='axis_translation_affine_static_v1'
        return contract
    # This is a synthetic historical-identity checkpoint at an open pose where
    # the correction changes no response, not a replay of consumed evidence.
    with monkeypatch.context() as patch:
        if moving:patch.setattr(TranslatingHemisphereCylinderManager,'restart_contract',old_path)
        else:patch.setattr(HemisphereSphereManager,'restart_contract',old_radius)
        prefix=solve_static_nonlinear(model,load,**common,max_load_factor=.25)
        assert prefix.status=='completed',prefix.failure_reason
        checkpoint=json.loads(json.dumps(prefix.to_restart_checkpoint(),allow_nan=False))
    fresh,_,_,_=factory()
    with pytest.raises(ValueError,match='analysis contract is incompatible'):
        solve_static_nonlinear(fresh,load,**common,restart_checkpoint=checkpoint)
