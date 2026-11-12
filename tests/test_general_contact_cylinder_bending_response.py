"""Small-motion cylinder cantilever discriminator with independent tip work.

The large-radius geometry differs explicitly from the plane continuum oracle;
this development fixture is not arbitrary-cylinder or finite-rotation acceptance.
"""
import json,os,sys
from fractions import Fraction
from pathlib import Path
import numpy as np
import pytest
from scipy.integrate import quad
from numpy.polynomial import Polynomial
from anyfesolver import (BoundaryCondition,ContactDefinition,ContactInteraction,
    ContactSolverSettings,ContactSurface,RigidContactBody,LoadCase,solve_static_nonlinear)
from anyfesolver.fe_core import FEModel
from anyfesolver.elements import BeamElement,QuadraticBeamElement

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'docs/d1_review_support'))
from d3_beam_foundation_reference import cantilever_foundation_reference
from test_general_contact_beam_foundation_dev import (RADIUS,YOUNG,NU,LENGTH,CLEARANCE,
    LOAD,AREA,INERTIA,BENDING,SHEAR,decimal_transfer_control)

LINE=1e4; ENDPOINT=10.; OBSTACLE_RADIUS=1e7


def fixture(quadratic,cells,line_stiffness=LINE,endpoint_stiffness=ENDPOINT):
    model=FEModel('free bending cylinder cantilever')
    model.add_material('elastic',YOUNG,NU,density=1.)
    count=2*cells+1 if quadratic else cells+1
    for node,x in enumerate(np.linspace(0.,LENGTH,count),1):model.add_node(node,x,0.,0.)
    weights=np.zeros(count)
    for i in range(cells):
        nodes=(2*i+1,2*i+2,2*i+3) if quadratic else (i+1,i+2)
        model.add_element(i+1,(QuadraticBeamElement if quadratic else BeamElement)(i+1,
            nodes,'elastic',{'area':AREA,'Iy':INERTIA,'Iz':INERTIA,'J':2*INERTIA}))
        for node,weight in zip(nodes,(1/6,2/3,1/6) if quadratic else (1/2,1/2)):
            weights[node-1]+=weight*LENGTH/cells
    model.add_boundary_condition(BoundaryCondition('root',(1,),
        {dof:0. for dof in ('ux','uy','uz','rx','ry','rz')}))
    model.add_boundary_condition(BoundaryCondition('planar',tuple(range(2,count+1)),
        {dof:0. for dof in ('uy','rx','rz')}))
    surface=ContactSurface('tube',tuple(range(1,cells+1)),beam_radius=RADIUS,beam_end_caps='hemisphere')
    body=RigidContactBody('cylinder','cylinder',(.5,0.,-OBSTACLE_RADIUS-RADIUS-CLEARANCE),
        normal=(0.,1.,0.),radius=OBSTACLE_RADIUS)
    law=ContactInteraction('tube_cylinder','tube','cylinder',None,1e-3*RADIUS,
        line_stiffness=line_stiffness,endpoint_stiffness=endpoint_stiffness)
    definition=ContactDefinition((surface,),(body,),(law,))
    settings=ContactSolverSettings(pair_motion='general',beam_end_cap_work='hemisphere_support_dev',
        line_integration_relative_tolerance=1e-7,maximum_line_subdivisions=14,
        maximum_swept_subdivisions=16)
    load=LoadCase('consistent uniform dead load')
    for node,weight in enumerate(weights,1):load.add_nodal_load(node,forces=(0.,0.,-LOAD*weight))
    return model,definition,settings,load


def test_independent_endpoint_reference_boundary_and_wrench():
    ref=cantilever_foundation_reference(length=LENGTH,clearance=CLEARANCE,bending=BENDING,
        shear=SHEAR,load=LOAD,line_stiffness=LINE,endpoint_stiffness=ENDPOINT)
    tip=ref.state(LENGTH)
    assert tip[0]>CLEARANCE and abs(tip[2])<1e-12
    cap=ENDPOINT*(tip[0]-CLEARANCE)
    assert abs(tip[3]+cap)<1e-12
    line=quad(lambda x:LINE*(ref.state(x)[0]-CLEARANCE),ref.front,LENGTH,
        epsabs=1e-14,epsrel=1e-11)[0]
    assert line+cap==pytest.approx(ref.reaction,rel=1e-9)
    assert cap>.01*ref.reaction and ref.contact_velocity_lower>0.


@pytest.mark.parametrize('factor',(1.,10.))
def test_endpoint_reference_decimal_resolution(factor):
    ref=cantilever_foundation_reference(length=LENGTH,clearance=CLEARANCE,bending=BENDING,
        shear=SHEAR,load=LOAD,line_stiffness=LINE*factor,endpoint_stiffness=ENDPOINT*factor)
    control=decimal_transfer_control(ref)
    record=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if record:Path(record).with_name(f'endpoint_reference_{factor:g}.json').write_text(
        json.dumps(control,indent=2),encoding='utf-8')


@pytest.mark.parametrize('quadratic',(False,True))
@pytest.mark.parametrize('cells',(32,64))
def test_coupled_bending_cylinder_against_endpoint_continuum(quadratic,cells):
    solve_case(quadratic,cells)


def pressure_errors(model,result,body,law,ref):
    """Direct polynomial line reaction, using a cancellation-free radial gap."""
    radius=body.radius+RADIUS
    radius_roundoff=float(Fraction(radius)-Fraction(body.radius)-Fraction(RADIUS))
    independent_minimum_gap=float('inf')
    offset=float(-Fraction(body.point[2])-Fraction(radius))
    difference_l1=0.;difference_l2=0.;reference_l2=0.;actual_integral=0.;quad_errors=np.zeros(4)
    actual_active=0.;positions_by_cell=[]
    for element in model.mesh.elements.values():
        reference=np.array([model.mesh.nodes[n].coords() for n in element.node_ids])
        positions=np.array([p+result.displacements[model.mesh.dof_manager.get_node_dofs(n)[:3]]
            for p,n in zip(reference,element.node_ids)])
        positions_by_cell.append(positions.tolist())
        s=Polynomial((0.,1.))
        shapes=(1-s,s) if len(positions)==2 else ((1-s)*(1-2*s),4*s*(1-s),s*(2*s-1))
        dx=sum((shape*position[0] for shape,position in zip(shapes,positions)),Polynomial((-body.point[0],)))
        delta=sum((shape*position[2] for shape,position in zip(shapes,positions)),Polynomial((offset,)))
        # Stable signed square: no subtraction of two order-1e14 terms.
        square=dx*dx+delta*(2*radius+delta)
        # Physical radius sum now governs line work as well as gap admission.
        # Differentiating this cancellation-free polynomial finds every possible
        # minimum of distance along the represented B2/B3 centreline.
        physical_square=square+radius_roundoff*(2*radius-radius_roundoff)
        extrema=[0.,1.]+[float(r.real) for r in square.deriv().roots()
            if abs(r.imag)<1e-10 and 0.<r.real<1.]
        independent_minimum_gap=min(independent_minimum_gap,*[
            physical_square(t)/(np.hypot(dx(t),radius+delta(t))+radius-radius_roundoff)
            for t in extrema])
        cuts=[0.]+sorted(float(r.real) for r in physical_square.roots()
            if abs(r.imag)<1e-10 and 0.<r.real<1.)+[1.]
        length=reference[-1,0]-reference[0,0]
        actual_active+=length*sum(b-a for a,b in zip(cuts,cuts[1:]) if physical_square((a+b)/2)<0.)
        if reference[0,0]<ref.front<reference[-1,0]:
            cuts=sorted(set(cuts+[(ref.front-reference[0,0])/length]))
        def fields(t):
            x=reference[0,0]+length*t
            rho=np.hypot(dx(t),radius+delta(t))
            gap=physical_square(t)/(rho+radius-radius_roundoff)
            actual=-law.line_stiffness*min(gap,0.)
            expected=ref.line_stiffness*max(ref.state(x)[0]-CLEARANCE,0.)
            return actual,expected
        functions=(lambda a,b:abs(a-b),lambda a,b:(a-b)**2,lambda a,b:b*b,lambda a,b:a)
        totals=[]
        for measure,function in enumerate(functions):
            total=0.
            for a,b in zip(cuts,cuts[1:]):
                value,error=quad(lambda t:function(*fields(t))*length,a,b,
                    epsabs=1e-16,epsrel=1e-9,limit=100)
                total+=value
                quad_errors[measure]+=error
            totals.append(total)
        difference_l1+=totals[0];difference_l2+=totals[1]
        reference_l2+=totals[2];actual_integral+=totals[3]
    reference_line=ref.reaction-ref.endpoint_stiffness*(ref.state(LENGTH)[0]-CLEARANCE)
    assert reference_l2>quad_errors[2]
    l2=np.sqrt(difference_l2/reference_l2)
    upper=np.sqrt((difference_l2+quad_errors[1])/(reference_l2-quad_errors[2]))
    lower=np.sqrt(max(difference_l2-quad_errors[1],0.)/(reference_l2+quad_errors[2]))
    return dict(relative_l1=difference_l1/reference_line,
        relative_l2=l2,actual_line_reaction=actual_integral,
        reference_line_reaction=reference_line,actual_active_length=actual_active,
        reference_active_length=LENGTH-ref.front,
        quadrature_estimated_errors=dict(l1_N=quad_errors[0],difference_l2_N2_per_m=quad_errors[1],
            reference_l2_N2_per_m=quad_errors[2],actual_line_N=quad_errors[3]),
        relative_l1_estimated_integration_error=quad_errors[0]/reference_line,
        relative_l2_estimated_integration_error=max(upper-l2,l2-lower),
        independent_minimum_line_gap_m=independent_minimum_gap,
        positions=positions_by_cell,combined_radius_roundoff_m=float(Fraction(radius)-
            Fraction(body.radius)-Fraction(RADIUS)),effective_clearance_m=float(
            -Fraction(body.point[2])-Fraction(body.radius)-Fraction(RADIUS)))


def solve_case(quadratic,cells,line_stiffness=LINE,endpoint_stiffness=ENDPOINT,inspect_pressure=False):
    model,definition,settings,load=fixture(quadratic,cells,line_stiffness,endpoint_stiffness)
    result=solve_static_nonlinear(model,load,contact=definition,contact_settings=settings,
        num_steps=4,max_iterations=20,tolerance=1e-10)
    assert result.status=='completed',result.failure_reason
    ref=cantilever_foundation_reference(length=LENGTH,clearance=CLEARANCE,bending=BENDING,
        shear=SHEAR,load=LOAD,line_stiffness=line_stiffness,endpoint_stiffness=endpoint_stiffness)
    info=result.info['general_contact'];tip_node=max(model.mesh.nodes)
    tip=-result.displacements[model.mesh.dof_manager.get_node_dofs(tip_node)[2]]
    contact=np.asarray(info['beam_force'])[2]
    errors=dict(tip=abs(tip-ref.state(LENGTH)[0])/ref.state(LENGTH)[0],
        reaction=abs(contact-ref.reaction)/ref.reaction)
    assert max(errors.values())<=1e-3,errors
    assert info['gap_tolerance_met'] and info['active_endpoints']==1
    assert info['candidate_endpoints']==2
    active=[field for field in info['endpoint_fields'] if field['active']]
    assert len(active)==1 and active[0]['node']==tip_node and active[0]['branch']=='sphere'
    root=np.asarray(result.steps[-1].support_reactions['root'])
    force=root[:3]+np.asarray(info['beam_force'])+(0.,0.,-LOAD*LENGTH)
    # Consistent nodal loads integrate x exactly, including authoritative axial motion.
    external_moment=np.zeros(3)
    for node,item in load.nodal_loads.items():
        position=model.mesh.nodes[node].coords()+result.displacements[model.mesh.dof_manager.get_node_dofs(node)[:3]]
        external_moment+=np.cross(position,item[:3])+item[3:]
    moment=root[3:]+np.asarray(info['beam_moment'])+external_moment
    assert np.linalg.norm(force)/(LOAD*LENGTH)<=1e-6
    assert np.linalg.norm(moment)/(LOAD*LENGTH**2)<=1e-6
    # Physical model mismatch bound at nominal stations; not a fit to solver output.
    geometry_bound=(LENGTH/2)**2/(2*OBSTACLE_RADIUS)
    assert geometry_bound/CLEARANCE<=1.25e-5
    pressure=pressure_errors(model,result,definition.rigid_bodies[0],definition.interactions[0],ref) if inspect_pressure else None
    if pressure is not None:
        assert max(pressure['relative_l1'],pressure['relative_l2'])<=.02,{
            key:pressure[key] for key in ('relative_l1','relative_l2')}
        assert pressure['actual_active_length']==pytest.approx(info['active_line_length'],rel=1e-6,abs=1e-12)
        assert pressure['independent_minimum_line_gap_m']>=-1e-5
        assert pressure['relative_l1_estimated_integration_error']<=.002
        assert pressure['relative_l2_estimated_integration_error']<=.002
        # The active tip sphere's minimum equals the line endpoint minimum in
        # this verified branch; reported outward direction must contain it.
        tip_positions=np.asarray(pressure['positions'][-1])
        derivative=(tip_positions[-1]-tip_positions[0] if len(tip_positions)==2
            else tip_positions[0]-4*tip_positions[1]+3*tip_positions[2])
        radial=tip_positions[-1]-np.asarray(definition.rigid_bodies[0].point)
        radial[1]=0.
        assert np.dot(derivative,-radial)>0.
    record=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if record:Path(record).with_name(f'bending_B{3 if quadratic else 2}_{cells}_{line_stiffness:g}_{endpoint_stiffness:g}.json').write_text(
        json.dumps(dict(cells=cells,quadratic=quadratic,tip=tip,reference_tip=ref.state(LENGTH)[0],
            reaction=contact,reference_reaction=ref.reaction,response_errors=errors,
            geometry_height_bound=geometry_bound,force_balance=force.tolist(),
            moment_balance=moment.tolist(),info=info,pressure=pressure,
            reference_path=str(Path(sys.modules['d3_beam_foundation_reference'].__file__).resolve())),
            indent=2),encoding='utf-8')
    return info,pressure


@pytest.mark.parametrize('quadratic',(False,True))
def test_line_reaction_resolution_and_stronger_enforcement(quadratic):
    baseline,_=solve_case(quadratic,64,LINE,ENDPOINT,True)
    stronger,_=solve_case(quadratic,64,10*LINE,10*ENDPOINT,True)
    assert abs(stronger['minimum_gap'])<abs(baseline['minimum_gap'])
