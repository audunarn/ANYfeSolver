"""Small producer/export/reader workflow; no new transient mechanical profile."""
import numpy as np
from anyfesolver import (FEModel,BoundaryCondition,ContactDefinition,ContactSurface,
    RigidContactBody,ContactInteraction,TransientConfig,ContactTransientSolverControls,
    LoadCase,create_shell_element,solve_transient_contact)
from anyfesolver.general_contact import read_transient_contact_results


def impact_archive():
    model=FEModel('archive-guided-q4-impact')
    model.add_material('elastic',1e5,.3,density=1.)
    for node,point in enumerate(((0.,0.,0.),(1.,0.,0.),(1.,1.,0.),(0.,1.,0.)),1):
        model.add_node(node,*point)
    model.add_element(1,create_shell_element(1,(1,2,3,4),'elastic',thickness=.05))
    model.add_boundary_condition(BoundaryCondition('guide',(1,2,3,4),
        {name:0. for name in ('ux','uy','rx','ry','rz')}))
    definition=ContactDefinition((ContactSurface('skin',(1,)),),
        (RigidContactBody('floor','plane',(0.,0.,-.01),(0.,0.,1.)),),
        (ContactInteraction('support','skin','floor',1e5,.001),))
    velocity=np.zeros(model.mesh.dof_manager.total_dofs)
    load=LoadCase('small-constant-downward')
    for node in (1,2,3,4):
        velocity[model.mesh.dof_manager.get_node_dofs(node)[2]]=-1.
        load.add_nodal_load(node,forces=(0.,0.,-.005))
    result=solve_transient_contact(model,definition,TransientConfig(.0001,.014,initial_velocity=velocity),
        base_load_case=load,solver_controls=ContactTransientSolverControls(save_constraint_reactions_dev=True),
        emit_restart_checkpoint=True)
    raw=result.to_archive_bytes()
    return result,read_transient_contact_results(raw)


if __name__=='__main__':
    from pathlib import Path
    result,archive=impact_archive()
    Path('transient_contact_results.archive.json').write_bytes(archive.to_bytes())
    print(dict(saved_states=len(result.times),sha256=archive.sha256,bytes=len(archive.to_bytes())))
