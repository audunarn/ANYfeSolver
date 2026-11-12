"""Bounded planar-reference Q4 finite-director, force-driven static path.

Twelve registered common rigid-motion segments precede free two-body bending
press/open/recontact. Physical virtual-spin forces are used with additive
rotation-chart Jacobians and the existing consistent corotational owner.
"""
from dataclasses import replace
import math
import numpy as np
from anyfesolver import BoundaryCondition, solve_static_nonlinear
from anyfesolver.general_contact import resolve_automatic_contact_domain
from anyfesolver.general_contact.static_path import (
    ContactPathControl, ContactPathNodalForce, ContactPathProgram,
)
from examples.general_contact_q4_pair_monolithic_cycle import build as base


def rotation(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array(((c, 0., s), (0., 1., 0.), (-s, 0., c)))


def rigid_pose(model, angle):
    q = np.zeros(model.mesh.dof_manager.total_dofs)
    r = rotation(angle)
    for node_id, node in model.mesh.nodes.items():
        dofs = model.mesh.dof_manager.get_node_dofs(node_id)
        p = np.array((node.x, node.y, node.z))
        q[dofs[:3]] = r@p-p
        q[dofs[3:6]] = (0., angle, 0.)
    return q


def build():
    model, domain, settings, _, _ = base()
    model.name = 'q4-finite-director-registered-rigid-motion-and-free-bending'
    model.clear_boundary_conditions()
    model.add_boundary_condition(BoundaryCondition('lower-y-anchor',(7,),{'uy':0.}))
    model.add_boundary_condition(BoundaryCondition('lower-yaw-anchor',(9,),{'uy':0.}))
    settings = replace(settings, shell_skin_kinematics='corotational_nodal_director_dev')
    domain = replace(domain, name='q4-finite-director-60deg-and-force-cycle')
    resolution = resolve_automatic_contact_domain(model, domain, settings,
                                                   analysis_control='contact_path')
    terms = [(node, name) for node in (1, 4)
             for name in ('ux','uy','uz','rx','ry','rz')]
    terms += [(node, 'uz') for node in (7,9,13,15)] + [(7,'ux')]
    controls = tuple(ContactPathControl.node(node, name,
        coordinate_profile='finite_shell_spin_chart_dev') for node, name in terms)
    angles = [math.pi*i/36. for i in range(13)] + [math.pi/3.]*3
    index = dict(ux=0,uy=1,uz=2,rx=3,ry=4,rz=5)
    waypoints = []
    for angle in angles:
        q = rigid_pose(model, angle)
        waypoints.append(tuple(float(q[model.mesh.dof_manager.get_node_dofs(node)[index[name]]])
                               for node, name in terms))
    normal = rotation(math.pi/3.)@np.array((0.,0.,1.))
    scalar = (0.,)*13 + (-5e-5,5e-5,-5e-5)
    forces = tuple(ContactPathNodalForce(node,name,
                   tuple(value*normal[axis] for value in scalar))
                   for node in (3,6) for axis,name in enumerate(('ux','uy','uz'))
                   if normal[axis] != 0.)
    program = ContactPathProgram(controls, tuple(waypoints), 1, forces,
                                 control_profile='finite_shell_triplets_dev')
    return model, domain, settings, resolution, program


def solve(model, settings, resolution, program, **kwargs):
    return solve_static_nonlinear(model, contact=resolution.definition,
        contact_settings=settings, contact_path_program=program,
        kinematics='corotational', corotational_tangent='consistent',
        tolerance=1e-10, max_iterations=18, emit_restart_checkpoint=True,
        record_increment_snapshots=True, **kwargs)


if __name__ == '__main__':
    import argparse,json,os
    from pathlib import Path
    parser=argparse.ArgumentParser()
    parser.add_argument('--through-step',type=int,choices=(1,4,8,12,13,14,15),default=4)
    parser.add_argument('--restart',type=str)
    args=parser.parse_args()
    model,domain,settings,resolution,program=build()
    restart=json.load(open(args.restart,encoding='utf-8')) if args.restart else None
    result=solve(model,settings,resolution,program,
                 max_load_factor=args.through_step/15.,restart_checkpoint=restart)
    report=dict(status=result.status,failure_reason=result.failure_reason,
        identity=resolution.identity,accepted_steps=[s.step_index for s in result.snapshots],
        capability=result.info.get('contact_path'),checkpoint=result.restart_checkpoint,
        element_state_coordinates=result.info.get('element_state_coordinates'),
        terminal_restart=bool(args.restart and not result.snapshots))
    encoded=json.dumps(report,default=lambda x:x.tolist() if hasattr(x,'tolist') else x)
    if os.environ.get('AUTOMATIC_DOMAIN_RESULT'):
        Path(os.environ['AUTOMATIC_DOMAIN_RESULT']).write_text(encoded,encoding='utf-8')
    print(encoded)
    assert result.status=='completed',result.failure_reason
