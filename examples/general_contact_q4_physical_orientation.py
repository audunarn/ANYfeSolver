"""Physical-side Q4/Q4 director declarations with free two-body bending."""
from dataclasses import replace
from anyfesolver import FEModel,create_shell_element,BoundaryCondition,solve_static_nonlinear
from anyfesolver.general_contact import resolve_automatic_contact_domain
from anyfesolver.general_contact.static_path import ContactPathProgram,ContactPathNodalForce
from examples.general_contact_automatic_q4_pair import build as declarations


def build(*, signs=(1,1), orientation="physical_sides_dev"):
    if signs not in {(1,1),(-1,1),(1,-1),(-1,-1)}:
        raise ValueError("two exact director signs required")
    _,domain,oldsettings,_,_=declarations(sliding=False,path_predictor='none')
    model=FEModel('q4-monolithic-positive-clearance-cycle');model.add_material('elastic',1e5,.3,density=1.)
    source=tuple((x,y,.04+2e-7) for y in (.1,1.1) for x in (.1,.6,1.1))
    target=tuple((x,y,-.01) for y in (0.,.5,1.) for x in (0.,.5,1.))
    for n,p in enumerate((*source,*target),1):model.add_node(n,*p)
    for eid,nodes in ((1,(1,2,5,4)),(2,(2,3,6,5)),(3,(7,8,11,10)),(4,(8,9,12,11)),(5,(10,11,14,13)),(6,(11,12,15,14))):
        sign=signs[0 if eid<=2 else 1]
        if sign<0:nodes=tuple(nodes[j] for j in (0,3,2,1))
        model.add_element(eid,create_shell_element(eid,nodes,'elastic',thickness=.05,reference_normal=(0.,0.,1.),director_polarity=sign))
    model.add_boundary_condition(BoundaryCondition('upper-left-clamp',(1,4),dict.fromkeys(('ux','uy','uz','rx','ry','rz'),0.)))
    model.add_boundary_condition(BoundaryCondition('lower-corner-normal',(7,9,13,15),{'uz':0.}))
    model.add_boundary_condition(BoundaryCondition('lower-inplane-anchor',(7,),{'ux':0.,'uy':0.}))
    model.add_boundary_condition(BoundaryCondition('lower-inplane-yaw-anchor',(9,),{'uy':0.}))
    participants=tuple(replace(p,participant=replace(p.participant,side=('bottom' if i==0 else 'top') if signs[i]>0 else ('top' if i==0 else 'bottom'))) for i,p in enumerate(domain.participants))
    domain=replace(domain,participants=participants,name='automatic-q4-physical-orientation-cycle',defaults=replace(domain.defaults,normal_law='augmented',normal_stiffness_n_per_m3=1000.))
    settings=replace(oldsettings,shell_pair_orientation=orientation,path_predictor='none',enforcement_solver='monolithic_active_set',complementarity_force_scale_n=1e-4,pressure_tolerance=1e-8)
    resolution=resolve_automatic_contact_domain(model,domain,settings,analysis_control='contact_path')
    program=ContactPathProgram(controls=(),waypoints=((),(),(),()),steps_per_segment=1,
        nodal_forces=tuple(ContactPathNodalForce(n,'uz',(0.,-5e-5,+5e-5,-5e-5)) for n in (3,6)))
    return model,domain,settings,resolution,program


def solve(model,settings,resolution,program,**kwargs):
    return solve_static_nonlinear(model,contact=resolution.definition,contact_settings=settings,
        contact_path_program=program,tolerance=1e-12,max_iterations=12,
        emit_restart_checkpoint=True,record_increment_snapshots=True,**kwargs)


if __name__=='__main__':
    import json
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--signs',choices=('pp','mp','pm','mm'),default='mm');args=parser.parse_args()
    signs=tuple(1 if x=='p' else -1 for x in args.signs)
    model,domain,settings,resolution,program=build(signs=signs);result=solve(model,settings,resolution,program)
    print(json.dumps(dict(status=result.status,failure_reason=result.failure_reason,identity=resolution.identity,
        director_signs=signs,orientation=settings.shell_pair_orientation,initial_clearance_m=2e-7,program=program.contract(),capability=result.info.get('contact_path'),
        accepted_steps=[s.step_index for s in result.snapshots],history=result.info.get('force_displacement_history')),
        default=lambda x:x.tolist() if hasattr(x,'tolist') else x))
    assert result.status=='completed',result.failure_reason

