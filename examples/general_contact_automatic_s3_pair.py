"""Automatic coarse-lower S3 pair with explicit physical skins and four path phases."""
from anyfesolver import BoundaryCondition,FEModel,create_shell_element,solve_static_nonlinear
from anyfesolver.general_contact import (ContactSurface,ContactSolverSettings,
    AutomaticContactParticipant,AutomaticContactDefaults,AutomaticContactDomain,
    resolve_automatic_contact_domain,verify_automatic_contact_resolution)


def solve_shell_pair(model,domain,settings,resolution,path,**controls):
    """Check the compiled reference contract immediately before execution."""
    definition=verify_automatic_contact_resolution(model,domain,resolution,settings)
    return solve_static_nonlinear(model,contact=definition,contact_settings=settings,
        contact_path_program=path,max_iterations=15,**controls)


def build_s3_pair(*,director_z=-1,resolve_domain=True):
    """Explicit downward directors preserve physical skins; lower coarse side integrates."""
    from anyfesolver import ContactPathControl,ContactPathProgram
    model=FEModel('declared-automatic-s3-pair')
    model.add_material('elastic',1e5,.3,density=1.)
    source=((.1,.1,.055),(.6,.1,.055),(1.1,.1,.055),
            (.1,1.1,.055),(.6,1.1,.055),(1.1,1.1,.055))
    target=((0.,0.,0.),(1.,0.,0.),(1.,1.,0.),(0.,1.,0.))
    for node,point in enumerate((*source,*target),1):model.add_node(node,*point)
    assert director_z in (-1,1)
    upper=((1,2,5),(1,5,4),(2,3,6),(2,6,5))
    lower=((7,8,10),(8,9,10))
    for element,nodes in enumerate((*upper,*lower),1):
        if director_z==-1:nodes=(nodes[0],nodes[2],nodes[1])
        model.add_element(element,create_shell_element(element,nodes,'elastic',thickness=.05,
            formulation='e4-pl-s3-v2d',reference_normal=(0.,0.,float(director_z)),
            reference_surface_offset=director_z*.01 if element>4 else 0.))
    model.add_boundary_condition(BoundaryCondition('source-guide',(1,2,3,4,5,6),
        {name:0. for name in ('uy','rx','ry','rz')}))
    model.add_boundary_condition(BoundaryCondition('target-guide',(7,8,9,10),
        {name:0. for name in ('ux','uy','rx','ry','rz')}))
    model.add_boundary_condition(BoundaryCondition('target-support',(7,8,10),{'uz':0.}))
    # One ordinary actuator coordinate with equal source transverse travel.
    actuator=model.mesh.dof_manager.get_node_dofs(1)[2]
    for node in range(2,7):
        dependent=model.mesh.dof_manager.get_node_dofs(node)[2]
        model.add_constraint_equation(terms=((dependent,1.),(actuator,-1.)),
            source_id=f'source-normal-actuator-{node}',dependent_dof=dependent)
    domain=AutomaticContactDomain('refined-projected-two-shells',(
        AutomaticContactParticipant(ContactSurface('upper',(1,2,3,4),'top' if director_z==-1 else 'bottom'),'upper-body'),
        AutomaticContactParticipant(ContactSurface('lower',(5,6),'bottom' if director_z==-1 else 'top'),'lower-body')),
        AutomaticContactDefaults(1000.,5e-5,normal_law='projected_penalty'),maximum_pairs=1)
    # Measured candidate settings; _dev suffixes mark unmeasured routes.
    settings=ContactSolverSettings(pair_motion='general',normal_discretization='primal_mortar',
        general_s3_differentiation='jet2_s3_dev',general_s3_difference_step=2e-5,
        q4_source_gap_certificate='whole_face',augmentation_support_resolution='exact',
        q4_source_area_tolerance=1e-9,q4_source_weighted_gap_tolerance=1e-10,
        q4_source_support_gradient_tolerance=1e-5,q4_source_support_hessian_tolerance=1e-5,
        q4_source_weighted_gap_gradient_tolerance=1e-5,q4_source_weighted_gap_hessian_tolerance=1e-5,
        maximum_q4_source_integration_triangles=128,path_predictor='control_projection_dev')
    resolution=resolve_automatic_contact_domain(model,domain,settings,
        analysis_control='contact_path') if resolve_domain else None
    # Five-micrometre nominal press retains the 50-micrometre swept gap guard.
    waypoints=((0.,0.),(0.,-.015005),(.02,-.015005),(.02,0.),(.02,-.015005))
    path=ContactPathProgram(controls=(ContactPathControl.node(1,'ux'),ContactPathControl.node(1,'uz')),
        waypoints=waypoints,steps_per_segment=1)
    return model,domain,settings,resolution,path



if __name__=='__main__':
    import json
    result=solve_shell_pair(*build_s3_pair(),emit_restart_checkpoint=True)
    if result.status!='completed':raise RuntimeError(result.failure_reason)
    print(json.dumps(dict(status=result.status,contact=result.info['general_contact'],phases=result.info['force_displacement_history']),default=lambda x:x.item() if hasattr(x,'item') else x.tolist()))
