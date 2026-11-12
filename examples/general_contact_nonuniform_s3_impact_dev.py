"""Two finite S3 bodies with nonuniform target motion at a common Newmark stage.

The source has uniform normal translation. The target retains translation and
an affine normal slope with fixed rotations: actual S3 shear work remains active.
This explicit fixed-projection development profile is not engineering acceptance.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from anyfesolver import (BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSurface, ContactSolverSettings, ContactTransientSolverControls, FEModel,
    TransientConfig, create_shell_element, solve_transient_contact)

GAP = .0002
KN = 1.e9
SUPPORT = np.array([1/3, 1/6, 1/3, 1/6])
X_BAR = np.array([.375, .75, .625, .25])-1/3
B = np.column_stack((np.ones(4), -np.ones(4), -X_BAR))
# The admitted S3 profile owns lumped translational mass: target nodal
# masses m/3 at x=(-2,5,-2), giving mean-square (x-1/3)^2=98/9.
MASS = np.diag([20., 49., 49.*98./9.])
MASS_XY = np.zeros((4,4))
MASS_XY[:2,:2] = np.diag([20.,49.])
MASS_XY[2:,2:] = 49.*np.array([[98.,-49.],[-49.,98.]])/9.
B_XY = np.column_stack((B, -(np.array([.375,.25,.625,.75])-1/3)))
INITIAL_VELOCITY = np.array([-.1, 0., 0.])
SETTINGS = ContactSolverSettings(pair_motion='general',
    general_s3_difference_step=1.e-5, general_s3_differentiation='jet2_s3_dev')
CONTROLS = ContactTransientSolverControls(allow_fixed_projection_s3_target_dev=True,
    q4_plane_event_subdivision_dev=True, batch_plane_event_fronts_dev=True,
    save_constraint_reactions_dev=True, equilibrium_tolerance=1.e-10)


def build_model(*, target_mode='x',target_xy=None,source_columns=1,normal_stiffness=KN):
    if target_mode not in ('x','xy'):
        raise ValueError('target_mode must be x or xy')
    if isinstance(source_columns,(bool,np.bool_)) or not isinstance(source_columns,(int,np.integer)) or source_columns<1:
        raise ValueError('source_columns must be a positive integer')
    columns=int(source_columns)
    model = FEModel('nonuniform-S3-target-impact-dev')
    model.add_material('source', 1.e7, .3, density=1000.)
    model.add_material('target', 1.e7, .3, density=100.)
    if target_xy is None:target_xy=((-2,-2),(5,-2),(-2,5))
    points=np.asarray(target_xy)
    if points.shape!=(3,2) or points.dtype.kind not in 'fiu' or not np.all(np.isfinite(points)):
        raise ValueError('target_xy must be three finite real planar points')
    source_xy=tuple((i/columns,0.) for i in range(columns+1))+tuple(
        (i/columns,1.) for i in range(columns,-1,-1))
    source_count=len(source_xy)
    coordinates=tuple((x,y,.0202) for x,y in source_xy)+tuple((point[0],point[1],0.) for point in points)
    for node, point in enumerate(coordinates, 1):
        model.add_node(node, *point)
    faces=[]
    for i in range(columns):
        bl,br,tr,tl=i+1,i+2,2*columns-i+1,2*columns-i+2
        faces.extend(([bl,br,tr],[bl,tr,tl]))
    for eid,nodes in enumerate((*faces,list(range(source_count+1,source_count+4))),1):
        model.add_element(eid, create_shell_element(eid, nodes,
            'source' if eid <= 2*columns else 'target', thickness=.02,
            formulation='e4-pl-s3-v2d', reference_normal=(0.,0.,1.)))
    model.add_boundary_condition(BoundaryCondition('normal-guides', list(range(1,source_count+4)),
        dict(ux=0., uy=0., rx=0., ry=0., rz=0.)))
    normal_dofs = [model.mesh.dof_manager.get_node_dofs(node)[2] for node in range(1,source_count+4)]
    for dof in normal_dofs[1:source_count]:
        model.add_constraint_equation(terms=((dof,1.), (normal_dofs[0],-1.)))
    if target_mode == 'x':
        model.add_constraint_equation(terms=((normal_dofs[source_count+2],1.), (normal_dofs[source_count],-1.)))
    modes = np.zeros((6*(source_count+3),3 if target_mode == 'x' else 4))
    for node, point in enumerate(coordinates, 1):
        dof = normal_dofs[node-1]
        if node <= source_count:
            modes[dof,0] = 1.
        else:
            modes[dof,1:] = ([1., point[0]-1/3] if target_mode == 'x'
                             else [1., point[0]-1/3, point[1]-1/3])
    definition = ContactDefinition(
        (ContactSurface('source', tuple(range(1,2*columns+1)), side='bottom'),
         ContactSurface('target', (2*columns+1,), side='top')), (),
        (ContactInteraction('pair', 'source', 'target', normal_stiffness, 2.e-5,
                            normal_law='projected_penalty'),))
    return model, definition, modes


def modal(values, modes):
    return np.asarray(values)@modes@np.linalg.inv(modes.T@modes)


def solve(*, dt=2.5e-5, end=.003, checkpoint=None, emit=False,
          controls=CONTROLS, cancellation_token=None, target_mode='x'):
    model, definition, modes = build_model(target_mode=target_mode)
    velocity = INITIAL_VELOCITY if target_mode == 'x' else np.r_[INITIAL_VELOCITY,0.]
    result = solve_transient_contact(model, definition,
        TransientConfig(dt, end, initial_velocity=modes@velocity),
        contact_settings=SETTINGS, solver_controls=controls,
        restart_checkpoint=checkpoint, emit_restart_checkpoint=emit,
        cancellation_token=cancellation_token)
    return model, result, modes


def run():
    _, result, modes = solve()
    q, v = modal(result.displacements, modes), modal(result.velocities, modes)
    return dict(status='development; not engineering acceptance', profile=result.profile,
        times_s=result.times.tolist(), q_m=q.tolist(), v_m_s=v.tolist(),
        contact=[row.to_dict() for row in result.contact],
        swept_minimum_gaps_m=result.swept_minimum_gaps.tolist(),
        cutbacks=result.cutbacks, events_s=result.localized_contact_event_times,
        cutback_diagnostics=result.cutback_diagnostics,
        endpoint_requests=result.event_endpoint_evaluations,
        stage_evaluations=result.plane_event_stage_evaluations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    text = json.dumps(run(), indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(text+'\n', encoding='utf-8')
    else:
        print(text)
