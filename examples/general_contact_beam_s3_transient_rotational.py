"""Numerical descriptor impact with actual affine eccentric target rotations."""
import numpy as np
from anyfesolver import BoundaryCondition
from anyfesolver.dynamics import TransientConfig
from anyfesolver.general_contact.transient import solve_transient_contact,ContactTransientSolverControls
from examples.general_contact_beam_s3_transient import build as base_build


def build():
    model,definition,settings,velocity=base_build(quadratic=True,unequal=True,flexural=True)
    model.boundary_conditions=[bc for bc in model.boundary_conditions if bc.name!='skin_director_guides']
    model.add_boundary_condition(BoundaryCondition('physical_skin_inplane_drill',[3,4,5],dict.fromkeys(('ux','uy','rz'),0.)))
    dof=lambda node: model.mesh.dof_manager.get_node_dofs(node)
    for axis in (3,4):
        for node in (4,5):
            model.add_constraint_equation(terms=((dof(node)[axis],1.),(dof(3)[axis],-1.)),dependent_dof=dof(node)[axis],source_id=f'skin_rotation_{axis}_{node}')
    # Exact planar normal-rate slope: zero initial MIN3 shear/curvature.
    gradient=np.array((-1.e-3/3,1.e-3/3))
    for node in (3,4,5):velocity[dof(node)[3:5]]=(gradient[1],-gradient[0])
    return model,definition,settings,velocity


def run(*,dt=1.e-5,end=.0018,restart=None):
    m,d,s,v=build()
    result=solve_transient_contact(m,d,TransientConfig(dt,end,initial_velocity=v),contact_settings=s,
        solver_controls=ContactTransientSolverControls(allow_rotational_beam_s3_dev=True,save_constraint_reactions_dev=True),
        emit_restart_checkpoint=True,restart_checkpoint=restart)
    return m,result


if __name__=='__main__':
    _,result=run()
    print(dict(profile=result.profile,states=len(result.times),active=sum(x.active_line_length>0 for x in result.contact),final_gap=result.contact[-1].minimum_gap))
