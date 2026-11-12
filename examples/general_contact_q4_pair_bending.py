"""Force-loaded two-body Q4 bending with admitted linearized director skins."""
from dataclasses import replace
from anyfesolver import BoundaryCondition,LoadCase,NonlinearConvergenceSettings,solve_static_nonlinear
from anyfesolver.general_contact import resolve_automatic_contact_domain
from examples.general_contact_automatic_q4_pair import build as guided_build


def build():
    # Reuse declarations/settings only; rebuild geometry/supports in a newmodel.
    from anyfesolver import FEModel,create_shell_element
    _,domain,oldsettings,_,_=guided_build(sliding=False,path_predictor='none')
    model=FEModel('q4-pair-free-bending-force');model.add_material('elastic',1e5,.3,density=1.)
    source=tuple((x,y,.04) for y in (.1,1.1) for x in (.1,.6,1.1))
    target=tuple((x,y,-.01) for y in (0.,.5,1.) for x in (0.,.5,1.))
    for node,point in enumerate((*source,*target),1):model.add_node(node,*point)
    faces=((1,(1,2,5,4)),(2,(2,3,6,5)),(3,(7,8,11,10)),(4,(8,9,12,11)),(5,(10,11,14,13)),(6,(11,12,15,14)))
    for eid,nodes in faces:model.add_element(eid,create_shell_element(eid,nodes,'elastic',thickness=.05,reference_normal=(0.,0.,1.)))
    model.add_boundary_condition(BoundaryCondition('upper-left-clamp',(1,4),{n:0. for n in ('ux','uy','uz','rx','ry','rz')}))
    model.add_boundary_condition(BoundaryCondition('lower-corner-normal',(7,9,13,15),{'uz':0.}))
    model.add_boundary_condition(BoundaryCondition('lower-inplane-anchor',(7,),{'ux':0.,'uy':0.}))
    model.add_boundary_condition(BoundaryCondition('lower-inplane-yaw-anchor',(9,),{'uy':0.}))
    settings=replace(oldsettings,path_predictor='none')
    resolution=resolve_automatic_contact_domain(model,domain,settings,analysis_control='force')
    load=LoadCase('dead-upper-end-force')
    for node in (3,6):load.add_nodal_load(node,forces=(0.,0.,-5e-5))
    return model,domain,settings,resolution,load


def solve(model,settings,resolution,load,**kwargs):
    return solve_static_nonlinear(model,load,contact=resolution.definition,contact_settings=settings,
        control='force',kinematics='von_karman',tolerance=1e-12,num_steps=2,max_iterations=12,min_step_fraction=.1,
        convergence_settings=NonlinearConvergenceSettings(growth_factor=1.,max_step_factor=1.),
        emit_restart_checkpoint=True,record_increment_snapshots=True,**kwargs)

if __name__=='__main__':
    import json
    model,domain,settings,resolution,load=build();result=solve(model,settings,resolution,load)
    print(json.dumps(dict(status=result.status,failure_reason=result.failure_reason,identity=resolution.identity,
        displacements=result.displacements,contact=result.info.get('general_contact'),steps=len(result.steps)),
        default=lambda x:x.tolist() if hasattr(x,'tolist') else x))
    assert result.status=='completed',result.failure_reason
