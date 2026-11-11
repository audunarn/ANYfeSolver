"""Automatic coarser S3/deformable Q4 press, 20mm actuator slide, open and recontact."""
from anyfesolver import BoundaryCondition,FEModel,create_shell_element,solve_static_nonlinear
from anyfesolver.general_contact import (ContactSurface,ContactSolverSettings,
    AutomaticContactParticipant,AutomaticContactDefaults,AutomaticContactDomain,
    resolve_automatic_contact_domain,verify_automatic_contact_resolution)


def solve_shell_pair(model,domain,settings,resolution,path,**controls):
    """Check the compiled reference contract immediately before execution."""
    definition=verify_automatic_contact_resolution(model,domain,resolution,settings)
    return solve_static_nonlinear(model,contact=definition,contact_settings=settings,
        contact_path_program=path,max_iterations=15,**controls)


def build(*,lifecycle=True,path_predictor='control_projection_dev',sliding=True):
    """Coarser two-S3 / one-Q4 projected-penalty development fixture.

    ``lifecycle=True`` presses, separates and presses again. All four source
    normal coordinates share the declared actuator displacement; the target
    retains its free normal coordinate. The projected weak-gap law reports
    regularization storage separately from contact work.
    ``sliding=True`` inserts a declared 20mm lateral actuator stage under
    press, then separates and recontacts at that lateral position. It requires
    the explicit development predictor; other three source ux remain unknowns.
    """
    from anyfesolver import ContactPathControl,ContactPathProgram
    if sliding and path_predictor!='control_projection_dev':
        raise ValueError('sliding example requires explicit control_projection_dev')
    model=FEModel('declared-automatic-coarser-s3-deformable-q4')
    model.add_material('elastic',1e5,.3,density=1.)
    source=((- .2,-.15,.065),(1.8,-.15,.065),(1.8,1.85,.065),(-.2,1.85,.065))
    target=((0.,0.,-.01),(1.,0.,-.01),(1.,1.,-.01),(0.,1.,-.01))
    for node,point in enumerate((*source,*target),1):model.add_node(node,*point)
    for element,nodes in ((1,(1,2,3)),(2,(1,3,4))):
        model.add_element(element,create_shell_element(element,nodes,'elastic',thickness=.05,
            formulation='e4-pl-s3-v2d',reference_normal=(0.,0.,1.),reference_surface_offset=.01))
    model.add_element(3,create_shell_element(3,(5,6,7,8),'elastic',thickness=.05,reference_normal=(0.,0.,1.)))
    model.add_boundary_condition(BoundaryCondition('source-guide',(1,2,3,4),
        {name:0. for name in ('uy','rx','ry','rz')}))
    model.add_boundary_condition(BoundaryCondition('target-guide',(5,6,7,8),
        {name:0. for name in ('ux','uy','rx','ry','rz')}))
    model.add_boundary_condition(BoundaryCondition('target-support',(5,6,8),{'uz':0.}))
    # One ordinary actuator coordinate with equal source transverse travel.
    actuator=model.mesh.dof_manager.get_node_dofs(1)[2]
    for node in range(2,5):
        dependent=model.mesh.dof_manager.get_node_dofs(node)[2]
        model.add_constraint_equation(terms=((dependent,1.),(actuator,-1.)),
            source_id=f'source-normal-actuator-{node}',dependent_dof=dependent)
    domain=AutomaticContactDomain('refined-projected-two-shells',(
        AutomaticContactParticipant(ContactSurface('s3_source',(1,2),'bottom'),'s3-body'),
        AutomaticContactParticipant(ContactSurface('q4_target',(3,),'top'),'q4-target-body')),
        AutomaticContactDefaults(1000.,5e-5,normal_law='projected_penalty'),maximum_pairs=1)
    settings=ContactSolverSettings(pair_motion='general',normal_discretization='primal_mortar',
        general_s3_differentiation='jet2_s3_q4_dev',general_s3_difference_step=2e-5,
        q4_source_gap_certificate='whole_face',augmentation_support_resolution='exact',
        q4_source_area_tolerance=1e-9,q4_source_weighted_gap_tolerance=1e-10,
        q4_source_support_gradient_tolerance=1e-5,q4_source_support_hessian_tolerance=1e-5,
        q4_source_weighted_gap_gradient_tolerance=1e-5,q4_source_weighted_gap_hessian_tolerance=1e-5,
        q4_target_weighted_gap_tolerance=1e-10,maximum_q4_target_integration_triangles=128,maximum_candidate_pairs=2,maximum_q4_source_integration_triangles=128,path_predictor=path_predictor)
    resolution=resolve_automatic_contact_domain(model,domain,settings,
        analysis_control='contact_path' if path_predictor!='none' else 'displacement')
    # Five-micrometre nominal press retains the 50-micrometre swept gap guard.
    travel=(0.,-.015005,0.,-.015005) if lifecycle else (0.,-.015005)
    waypoints=((0.,0.),(0.,-.015005),(.02,-.015005),(.02,0.),(.02,-.015005)) if sliding else tuple((0.,value) for value in travel)
    path=ContactPathProgram(controls=(ContactPathControl.node(1,'ux'),ContactPathControl.node(1,'uz')),
        waypoints=waypoints,steps_per_segment=1 if lifecycle or sliding else 2)
    return model,domain,settings,resolution,path


if __name__=='__main__':
    import json
    model,domain,settings,resolution,path=build()
    result=solve_shell_pair(model,domain,settings,resolution,path,emit_restart_checkpoint=True)
    print(json.dumps(dict(status=result.status,failure_reason=result.failure_reason,
        identity=resolution.identity,selections=[x.to_dict() if hasattr(x,'to_dict') else str(x) for x in resolution.selections],
        contact=result.info.get('general_contact'),phases=result.info.get('force_displacement_history')),
        default=lambda x:x.item() if hasattr(x,'item') else x.tolist()))
    assert result.status=='completed',result.failure_reason
